"""Bibliothèque de modèles de projection simples, appliqués rubrique par rubrique (budget et réalisé séparément).

Chaque modèle reçoit la série mensuelle d'une rubrique et renvoie les mois suivants :

- linear   : tendance linéaire (moindres carrés) sur les totaux annuels, répartie selon la saisonnalité ;
- cagr     : croissance composée (taux annuel moyen) sur les totaux annuels, même répartition ;
- snaive   : naïf saisonnier, chaque mois reprend la valeur du même mois de l'année précédente ;
- ar       : autorégression y_t = c + a·y(t-1) + b·y(t-12), estimée par moindres carrés, projetée pas à pas ;
- rf       : forêt aléatoire « globale » par bloc, sur l'écart annuel y_t - y(t-12) normalisé
             (variables : mois, y(t-12), rubrique) ;
- ensemble : moyenne des cinq modèles ci-dessus ;
- auto     : pour chaque rubrique, le modèle qui s'est le moins trompé au test rétrospectif.

La « plage » d'une projection est l'écart entre le plus bas et le plus haut des modèles de tendance
(linéaire, composée, AR, forêt aléatoire) ; le naïf saisonnier, sans croissance, sert de référence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from budget_data import MONTHS, REALISED

MODELS = {
    "auto": "Meilleur modèle (auto)",
    "linear": "Tendance linéaire",
    "cagr": "Croissance composée",
    "snaive": "Naïf saisonnier",
    "ar": "Autorégression (AR)",
    "rf": "Forêt aléatoire",
    "ensemble": "Moyenne des modèles",
}
DESCRIPTIONS = {
    "auto": "pour chaque rubrique, le modèle dont l'écart mois par mois a été le plus faible au test "
            "rétrospectif (à égalité, le plus simple)",
    "linear": "droite des moindres carrés sur les totaux annuels, répartie sur les mois selon la saisonnalité "
              "habituelle de la rubrique",
    "cagr": "taux de croissance annuel moyen appliqué à la dernière année, même répartition mensuelle",
    "snaive": "chaque mois reprend la valeur du même mois de l'année précédente (aucune croissance)",
    "ar": "y(t) = c + a·y(t-1) + b·y(t-12), estimé par moindres carrés puis prolongé mois par mois",
    "rf": "une forêt de 100 arbres, apprise sur toutes les rubriques du bloc, prédit l'écart avec le même "
          "mois de l'an dernier à partir du mois, de y(t-12) et de la rubrique",
    "ensemble": "moyenne simple des cinq autres modèles",
}
BASE_MODELS = ["linear", "cagr", "snaive", "ar", "rf"]
CANDIDATES = BASE_MODELS + ["ensemble"]  # modèles calculés directement ; « auto » choisit parmi eux
RANGE_MODELS = ["linear", "cagr", "ar", "rf"]  # la plage ignore le naïf (sans croissance) et la moyenne
KEYS = ["scenario", "block", "category"]


# --------------------------------------------------------------------------- modèles

def project(years, values, targets, method: str) -> np.ndarray:
    """Projection annuelle : tendance linéaire ou croissance composée."""
    years = np.asarray(years, float)
    values = np.asarray(values, float)
    targets = np.asarray(targets, float)
    if len(values) == 0:
        return np.zeros(len(targets))
    if len(values) == 1 or np.ptp(years) == 0:
        return np.full(len(targets), values[-1])
    if method == "cagr" and values[0] > 0 and values[-1] > 0:
        growth = (values[-1] / values[0]) ** (1 / (years[-1] - years[0]))
        return values[-1] * growth ** (targets - years[-1])
    slope, intercept = np.polyfit(years, values, 1)
    return np.maximum(intercept + slope * targets, 0.0)


def _phasing(y: np.ndarray) -> np.ndarray:
    years = y.reshape(-1, 12)
    totals = years.sum(axis=1)
    ok = totals > 0
    if not ok.any():
        return np.full(12, 1 / 12)
    share = (years[ok] / totals[ok, None]).mean(axis=0)
    return share / share.sum()


def _annual(y: np.ndarray, n_years: int, method: str) -> np.ndarray:
    totals = y.reshape(-1, 12).sum(axis=1)
    years = np.arange(len(totals))
    annual = project(years, totals, np.arange(len(totals), len(totals) + n_years), method)
    return (annual[:, None] * _phasing(y)[None, :]).reshape(-1)


def _snaive(y: np.ndarray, h: int) -> np.ndarray:
    return np.tile(y[-12:], int(np.ceil(h / 12)))[:h]


def _ar(y: np.ndarray, h: int) -> np.ndarray:
    if len(y) < 24:
        return _snaive(y, h)
    t = np.arange(12, len(y))
    X = np.column_stack([np.ones(len(t)), y[t - 1], y[t - 12]])
    scale = max(float(np.abs(y).max()), 1.0)
    Xs, ys = X / np.array([1.0, scale, scale]), y[t] / scale
    coef = np.linalg.solve(Xs.T @ Xs + 1e-6 * np.eye(3), Xs.T @ ys)  # légère régularisation
    hist = list(y / scale)
    for _ in range(h):
        hist.append(max(coef[0] + coef[1] * hist[-1] + coef[2] * hist[-12], 0.0))
    return np.array(hist[len(y):]) * scale


def _rf_group(series: list[np.ndarray], years_ahead: list[int]) -> list[np.ndarray]:
    """Forêt aléatoire « globale » : un seul modèle appris sur toutes les rubriques d'un même bloc.

    Les montants sont normalisés par la moyenne de chaque rubrique ; le modèle prédit l'écart avec le
    même mois de l'an dernier à partir du mois, de y(t-12) et de l'identifiant de la rubrique.
    """
    scales = [float(y.mean()) if y.mean() > 0 else 1.0 for y in series]
    X, target = [], []
    for i, (y, s) in enumerate(zip(series, scales)):
        if len(y) >= 24:
            t = np.arange(12, len(y))
            X.append(np.column_stack([t % 12, y[t - 12] / s, np.full(len(t), i)]))
            target.append((y[t] - y[t - 12]) / s)
    if not X:
        return [_snaive(y, 12 * n) for y, n in zip(series, years_ahead)]
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, random_state=0, n_jobs=1)
    model.fit(np.vstack(X), np.concatenate(target))
    hists = [np.asarray(y, float) for y in series]
    for k in range(max(years_ahead, default=0)):  # une année entière à la fois : y(t-12) est connu
        live = [i for i, n in enumerate(years_ahead) if n > k]
        rows = np.vstack([np.column_stack([np.arange(12), hists[i][-12:] / scales[i], np.full(12, i)])
                          for i in live])
        step = model.predict(rows).reshape(len(live), 12)
        for j, i in enumerate(live):
            hists[i] = np.concatenate([hists[i], np.maximum(hists[i][-12:] + step[j] * scales[i], 0.0)])
    return [h[len(y):] for h, y in zip(hists, series)]


def predict_all(y: np.ndarray, n_years: int, rf: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Toutes les projections mensuelles pour `n_years` années après une série d'années complètes.

    `rf` : projection de la forêt aléatoire, calculée au niveau du bloc par `_rf_group`.
    """
    h = 12 * n_years
    y = np.nan_to_num(np.asarray(y, float))
    if h == 0:
        return {m: np.zeros(0) for m in CANDIDATES}
    out = {
        "linear": _annual(y, n_years, "linear"),
        "cagr": _annual(y, n_years, "cagr"),
        "snaive": _snaive(y, h),
        "ar": _ar(y, h),
        "rf": rf if rf is not None else _rf_group([y], [n_years])[0],
    }
    out["ensemble"] = np.mean([out[m] for m in BASE_MODELS], axis=0)
    return out


# --------------------------------------------------------------------------- application aux données

def _series(long: pd.DataFrame) -> dict[tuple, list]:
    """Regroupe par (scénario, bloc) la liste des (rubrique, années, matrice année x mois)."""
    groups: dict[tuple, list] = {}
    for (scenario, block, category), g in long.groupby(KEYS, sort=False):
        grid = g.pivot_table(index="year", columns="month", values="amount", aggfunc="sum")
        grid = grid.reindex(columns=MONTHS).fillna(0)
        groups.setdefault((scenario, block), []).append((category, grid.index.to_numpy(int), grid.to_numpy(float)))
    return groups


def last_actual_year(long: pd.DataFrame) -> int:
    realised = long.loc[long["scenario"] == REALISED, "year"]
    return int(realised.max() if len(realised) else long["year"].max())


def forecast_all(long: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Projections mensuelles des modèles calculés pour les `horizon` années après la dernière année réalisée.

    Une année déjà présente dans le classeur (par ex. un budget de l'année à venir) est reprise telle quelle.
    Le modèle « auto » s'ajoute ensuite avec `add_auto`, à partir du test rétrospectif.
    """
    last = last_actual_year(long)
    targets = list(range(last + 1, last + 1 + horizon))
    rows = []
    for (scenario, block), items in _series(long).items():
        aheads = [max(0, targets[-1] - int(years.max())) for _, years, _ in items]
        rf = _rf_group([grid.reshape(-1) for _, _, grid in items], aheads)
        for (category, years, grid), ahead, rf_pred in zip(items, aheads, rf):
            preds = predict_all(grid.reshape(-1), ahead, rf_pred)
            for model in CANDIDATES:
                for year in targets:
                    if year in years:
                        values, source = grid[list(years).index(year)], "file"
                    else:
                        k = year - int(years.max()) - 1
                        values, source = preds[model][12 * k: 12 * (k + 1)], "forecast"
                    rows.extend((model, scenario, block, category, year, m, float(v), source)
                                for m, v in zip(MONTHS, values))
    return pd.DataFrame(rows, columns=["model"] + KEYS + ["year", "month", "amount", "source"])


def combine(all_fc: pd.DataFrame, long: pd.DataFrame, method: str, adjust: float = 0.0) -> pd.DataFrame:
    """Projection du modèle choisi, avec la plage (min / max) couverte par les modèles de tendance."""
    ids = KEYS + ["year", "month"]
    spread = (all_fc[all_fc["model"].isin(RANGE_MODELS)].groupby(ids, sort=False)["amount"]
              .agg(low="min", high="max").reset_index())
    out = all_fc[all_fc["model"] == method].drop(columns="model").merge(spread, on=ids)
    factor = np.where(out["source"] == "forecast", 1 + adjust, 1.0)
    for col in ("amount", "low", "high"):
        out[col] = out[col] * factor
    labels = long[["category", "label_fr", "label_en", "order"]].drop_duplicates("category")
    out = out.merge(labels, on="category", how="left")
    out["date"] = pd.to_datetime(pd.DataFrame({"year": out["year"], "month": out["month"], "day": 1}))
    return out


def best_models(bt: pd.DataFrame) -> pd.DataFrame:
    """Par rubrique, le modèle au plus faible écart absolu cumulé sur l'année de test (WAPE).

    Le WAPE (somme des écarts mensuels / total réel) reste défini pour les rubriques ponctuelles, qui ont
    des mois à zéro. À égalité (au millionième près), le modèle le plus simple l'emporte.
    """
    err = (bt[bt["model"].isin(CANDIDATES)]
           .assign(abs_err=lambda d: (d["predicted"] - d["actual"]).abs())
           .groupby(KEYS + ["model"], sort=False)[["abs_err", "actual"]].sum().reset_index())
    err["wape"] = err["abs_err"] / err["actual"].where(err["actual"] > 0)
    err["_tie"] = err["wape"].round(6)
    err["_rank"] = err["model"].map({m: i for i, m in enumerate(CANDIDATES)})
    best = err.sort_values(KEYS + ["_tie", "_rank"], na_position="last").drop_duplicates(KEYS)
    return best[KEYS + ["model", "wape"]].reset_index(drop=True)


def add_auto(all_fc: pd.DataFrame, bt: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le modèle « auto » : pour chaque rubrique, les lignes du modèle retenu par `best_models`.

    Une rubrique sans test possible (historique trop court) garde la tendance linéaire.
    """
    keys = all_fc[KEYS].drop_duplicates()
    choice = keys.merge(best_models(bt)[KEYS + ["model"]], on=KEYS, how="left").fillna({"model": "linear"})
    auto = all_fc.merge(choice, on=KEYS + ["model"]).assign(model="auto")
    return pd.concat([all_fc, auto], ignore_index=True)


def backtest_all(long: pd.DataFrame) -> pd.DataFrame:
    """Test rétrospectif : chaque modèle apprend sans la dernière année réalisée et doit l'estimer."""
    test = last_actual_year(long)
    rows = []
    for (scenario, block), items in _series(long).items():
        items = [(c, y, g) for c, y, g in items if (y < test).sum() >= 2 and test in y]
        if not items:
            continue
        trains = [grid[years < test].reshape(-1) for _, years, grid in items]
        rf = _rf_group(trains, [1] * len(items))
        for (category, years, grid), train, rf_pred in zip(items, trains, rf):
            preds = predict_all(train, 1, rf_pred)
            actual = grid[list(years).index(test)]
            for model in CANDIDATES:
                rows.extend((model, scenario, block, category, test, m, a, float(p))
                            for m, a, p in zip(MONTHS, actual, preds[model][:12]))
    return pd.DataFrame(rows, columns=["model"] + KEYS + ["year", "month", "actual", "predicted"])


def score(bt: pd.DataFrame) -> pd.DataFrame:
    """Erreur annuelle et erreur mensuelle moyenne (MAPE) par modèle, sur la sélection fournie."""
    rows = []
    for model, g in bt.groupby("model", sort=False):
        monthly = g.groupby("month")[["actual", "predicted"]].sum()
        actual_total = monthly["actual"].sum()
        nonzero = monthly["actual"] > 0
        mape = (np.abs(monthly["predicted"] - monthly["actual"])[nonzero] / monthly["actual"][nonzero]).mean()
        rows.append({"model": model, "label": MODELS[model],
                     "annual_error": monthly["predicted"].sum() / actual_total - 1 if actual_total else np.nan,
                     "mape": mape if nonzero.any() else np.nan})
    return pd.DataFrame(rows)


def trend_fit(long: pd.DataFrame) -> pd.DataFrame:
    """R² d'une droite passant par les totaux annuels, par bloc et scénario."""
    totals = long.groupby(["block", "scenario", "year"], as_index=False)["amount"].sum()
    rows = []
    for (block, scenario), g in totals.groupby(["block", "scenario"]):
        x, y = g["year"].to_numpy(float), g["amount"].to_numpy(float)
        r2 = np.nan
        if len(x) >= 3 and np.ptp(y) > 0:
            fitted = np.polyval(np.polyfit(x, y, 1), x)
            r2 = 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append({"block": block, "scenario": scenario, "r2": r2})
    return pd.DataFrame(rows)
