"""Tableau de bord des dépenses : investissement (CAPEX) et consommation (OPEX), par rubrique,
avec une projection simple et explicable (tendance annuelle x saisonnalité mensuelle).

Lancement :  python -m streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import charts as ch
import forecasting as fc
import ratios as rt
from budget_data import (BLOCKS, CAPEX, MONTH_ABBR, MONTH_NAMES, MONTHS, OPEX, PLANNED, REALISED, BudgetData,
                         load_workbook)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_FILE = APP_DIR / "claude test.xlsx"
METHOD_KEYS = {label: key for key, label in fc.MODELS.items()}
# Les projections sont mises en cache sur disque : la clé inclut le code des modèles, pour qu'une
# modification de forecasting.py ne serve jamais d'anciens résultats.
MODEL_VERSION = hashlib.sha1(Path(fc.__file__).read_bytes()).hexdigest()[:12]
SERIES = {REALISED: "Réalisé", PLANNED: "Budget"}
BLOCK_OF = {CAPEX: "d'investissement (CAPEX)", OPEX: "de consommation (OPEX)"}  # « des dépenses … »
BLOCK_TITLE = {CAPEX: "Investissement (CAPEX)", OPEX: "Consommation (OPEX)"}
ALL = "__all__"
BLOCK_TABS = {CAPEX: ":material/construction: Investissement (CAPEX)",
              OPEX: ":material/shopping_cart: Consommation (OPEX)"}
GAP_TAB = ":material/compare_arrows: Écarts budget / réalisé"
RATES_TAB = ":material/percent: Taux de gestion"
GAP_SCOPES = {ALL: "Tous les blocs", CAPEX: "Investissement (CAPEX)", OPEX: "Consommation (OPEX)"}
RATE_VIEWS = {"realisation": "Taux de réalisation", "consumption": "Consommation du budget",
              "growth": "Taux d'évolution", "structure": "Taux de structure"}
FILTER_KEYS = ("f_years", "f_months", "f_method", "f_horizon", "f_whatif", "f_series")
TAB_KEY_PREFIXES = ("capex_", "opex_", "gap_", "rates_")

st.set_page_config(page_title="Suivi des dépenses", page_icon=":material/monitoring:", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
[data-testid="stMainBlockContainer"] {padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1520px;}
[data-testid="stMetric"] {padding: .85rem 1rem .7rem;}
[data-testid="stMetricLabel"] p {font-weight: 550; opacity: .78;}
[class*="st-key-kpis"] {align-items: stretch !important;}
[class*="st-key-kpis"] > div {flex: 1 1 215px !important; min-width: 215px !important; width: auto !important;}
[class*="st-key-kpis"] [data-testid="stMetric"] {height: 100%;}
.card-title {font-weight: 650; font-size: 1.02rem; line-height: 1.3; margin: 0;}
.card-sub {opacity: .62; font-size: .84rem; line-height: 1.35; margin: .2rem 0 .35rem;}
[class*="st-key-insight"] {border-left: 3px solid #2a78d6 !important;}
[class*="st-key-insight"] ul {margin-bottom: 0;}
[class*="st-key-insight"] li {margin-bottom: .15rem;}
</style>
"""

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|~$])")


def md(text) -> str:
    """Échappe les caractères Markdown dans les textes issus des données."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def pct(value) -> str:
    return ch.pct(value, signed=True)


def model_phrase(method: str) -> str:
    """Nom du modèle à glisser dans une phrase (sans parenthèses imbriquées)."""
    return "sélection automatique" if method == "auto" else fc.MODELS[method].lower()


# --------------------------------------------------------------------------- filtres

@dataclass(frozen=True)
class Filters:
    years: tuple[int, ...]
    months: tuple[int, int]
    method: str
    horizon: int
    adjust: float
    unit: str
    currency: str
    series: str

    @property
    def month_list(self) -> list[int]:
        return list(range(self.months[0], self.months[1] + 1))

    @property
    def period(self) -> str:
        a, b = self.months
        if (a, b) == (1, 12):
            return "année complète"
        return MONTH_ABBR[a - 1] if a == b else f"{MONTH_ABBR[a - 1]}–{MONTH_ABBR[b - 1]}"

    @property
    def years_label(self) -> str:
        ys = sorted(self.years)
        if len(ys) > 1 and ys == list(range(ys[0], ys[-1] + 1)):
            return f"{ys[0]}–{ys[-1]}"
        return ", ".join(map(str, ys))

    @property
    def series_label(self) -> str:
        return SERIES[self.series]

    def money(self, value, signed: bool = False) -> str:
        return ch.money(value, self.unit, self.currency, signed)

    def tag(self, value, signed: bool = False) -> str:
        """Montant compact pour les étiquettes : la devise figure déjà dans le titre de l'axe."""
        return ch.money(value, self.unit, "", signed)

    def unit_for(self, magnitude) -> ch.Unit:
        magnitude = float(np.nanmax(np.abs(np.atleast_1d(magnitude)))) if np.size(magnitude) else 0.0
        return ch.choose_unit(self.unit, 0.0 if np.isnan(magnitude) else magnitude, self.currency)


def compares_both_series() -> bool:
    """Les pages Écarts et Taux utilisent toujours budget et réalisé ensemble."""
    return st.session_state.get("main_tab") in (GAP_TAB, RATES_TAB)


def reset_filters() -> None:
    """Oublie l'état des filtres : ils reviennent à leurs valeurs par défaut."""
    for key in list(st.session_state):
        if key in FILTER_KEYS or key.startswith(TAB_KEY_PREFIXES):
            del st.session_state[key]


def source_picker() -> tuple[bytes | None, str | None]:
    with st.sidebar:
        st.markdown("### :material/monitoring: Suivi des dépenses")
        upload = st.file_uploader(
            "Classeur", type=["xlsx", "xlsm"], key="upload",
            help="Facultatif. Même structure que « claude test.xlsx » : une feuille par Budget / Réalisation "
                 "et par année, un bloc CAPEX puis un bloc OPEX, une colonne annuelle et douze colonnes mensuelles.")
    if upload is not None:
        return upload.getvalue(), upload.name
    if DEFAULT_FILE.exists():
        return DEFAULT_FILE.read_bytes(), DEFAULT_FILE.name
    return None, None


def sidebar(data: BudgetData) -> Filters:
    years = data.years
    if st.session_state.get("_years_seen") != years:  # nouveau classeur : repartir de toutes ses années
        st.session_state["_years_seen"] = years
        st.session_state.pop("f_years", None)
    with st.sidebar:
        st.caption(f":material/description: **{md(data.source_name)}** · {len(data.sheets)} feuilles · "
                   f"{years[0]}–{years[-1]}")
        st.markdown("#### Filtres")
        series = st.segmented_control("Données", list(SERIES), format_func=SERIES.get, default=REALISED,
                                      key="f_series", required=True, disabled=compares_both_series(),
                                      help="Réalisé = feuilles « Realisation » ; Budget = feuilles « Budget ». "
                                           "Sans objet sur les pages Écarts et Taux, qui utilisent toujours les deux.")
        sel_years = st.pills("Années", years, selection_mode="multi", default=years, key="f_years")
        months = st.select_slider("Mois", options=MONTHS, value=(1, 12), key="f_months",
                                  format_func=lambda m: MONTH_ABBR[m - 1],
                                  help="Limite toutes les vues à une partie de l'année, par ex. janv.–sept. "
                                       "pour comparer des cumuls à date.")
        st.markdown("#### Projection")
        method = st.selectbox("Modèle", list(METHOD_KEYS), index=0, key="f_method",
                              help="\n".join(f"- **{fc.MODELS[k]}** : {fc.DESCRIPTIONS[k]}." for k in fc.MODELS)
                                   + "\n\nTous les modèles sont calculés ; comparez-les en bas de chaque onglet.")
        horizon = st.slider("Horizon (années)", 1, 3, value=1, key="f_horizon")
        whatif = st.slider("Ajustement de scénario", -20, 20, value=0, key="f_whatif", format="%+d%%",
                           help="Décale toutes les projections, par ex. +5 % pour un choc d'inflation.")
        with st.expander("Affichage", icon=":material/tune:"):
            unit = st.segmented_control("Montants en", ["Auto", "Md", "M"], default="Auto", key="f_unit",
                                        required=True, help="Md = milliards, M = millions.")
            currency = st.text_input("Devise", value="DZD", max_chars=8, key="f_currency")
        if len(data.repair_log):
            st.caption(f":material/build: {len(data.repair_log)} lignes mensuelles corrigées",
                       help="En 2024–2025, cinq rubriques CAPEX ponctuelles (Camions, Infrastructure, Matériel "
                            "audiovisuel, Ouvrages d'infrastructure, VHL) répètent le montant annuel chaque mois. "
                            "Le montant annuel est réparti comme l'année propre la plus proche, ce qui correspond "
                            "aux lignes de total du classeur.")
        st.button("Réinitialiser les filtres", icon=":material/restart_alt:", on_click=reset_filters,
                  width="stretch")
    return Filters(years=tuple(sorted(sel_years or [])), months=tuple(months), method=METHOD_KEYS[method],
                   horizon=int(horizon), adjust=whatif / 100, unit=unit, currency=(currency or "").strip(),
                   series=series)


# --------------------------------------------------------------------------- calculs mis en cache

@st.cache_data(show_spinner="Lecture du classeur…", max_entries=8)
def load(file_bytes: bytes, name: str) -> BudgetData:
    return load_workbook(file_bytes, name, "rephase")


MAX_HORIZON = 3


@st.cache_data(show_spinner="Test rétrospectif des modèles…", max_entries=8, persist="disk")
def model_checks(long: pd.DataFrame, version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fc.backtest_all(long), fc.trend_fit(long)


@st.cache_data(show_spinner="Calcul des modèles de projection…", max_entries=8, persist="disk")
def all_forecasts(long: pd.DataFrame, version: str) -> pd.DataFrame:
    """Tous les modèles, sur l'horizon maximal : changer de modèle ou d'horizon ne recalcule rien."""
    backtest, _ = model_checks(long, version)
    return fc.add_auto(fc.forecast_all(long, MAX_HORIZON), backtest)


@st.cache_data(show_spinner=False, max_entries=32)
def get_forecast(long: pd.DataFrame, method: str, horizon: int, adjust: float) -> pd.DataFrame:
    every = all_forecasts(long, MODEL_VERSION)
    every = every[every["year"] <= fc.last_actual_year(long) + horizon]
    return fc.combine(every, long, method, adjust)


# --------------------------------------------------------------------------- aides données

def scoped(df: pd.DataFrame, f: Filters, blocks=BLOCKS, categories=None, use_years: bool = True,
           both: bool = False) -> pd.DataFrame:
    """Applique les filtres ; `both=True` garde budget et réalisé (page des écarts)."""
    mask = df["block"].isin(blocks) & df["month"].between(*f.months)
    if not both:
        mask &= df["scenario"] == f.series
    if use_years:
        mask &= df["year"].isin(f.years)
    if categories:
        mask &= df["category"].isin(categories)
    return df[mask]


def year_total(df: pd.DataFrame, year: int) -> float:
    return float(df.loc[df["year"] == year, "amount"].sum())


def last_actual_year(data: BudgetData) -> int:
    return fc.last_actual_year(data.long)


def label_lookup(data: BudgetData) -> dict[str, str]:
    cats = data.long.drop_duplicates("category")
    return dict(zip(cats["category"], cats["label_fr"]))


def categories_for(data: BudgetData, block: str) -> list[str]:
    cats = data.long[data.long["block"] == block].drop_duplicates("category")
    return cats.sort_values("order")["category"].tolist()


def monthly_frame(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """Totaux mensuels sur un calendrier continu ; les mois hors filtre restent vides (trous)."""
    p = df.groupby("date")[["amount"]].sum().rename(columns={"amount": "value"})
    if p.empty:
        return p.reset_index()
    start = pd.Timestamp(year=min(f.years), month=f.months[0], day=1)
    end = pd.Timestamp(year=max(f.years), month=f.months[1], day=1)
    p = p.reindex(pd.date_range(start, end, freq="MS"))
    keep = p.index.year.isin(f.years) & (p.index.month >= f.months[0]) & (p.index.month <= f.months[1])
    p.loc[~keep] = np.nan
    return p.rename_axis("date").reset_index()


def comparable_years(data: BudgetData, f: Filters) -> list[int]:
    """Années sélectionnées qui ont à la fois une feuille Budget et une feuille Réalisation."""
    per_year = data.long.groupby("year")["scenario"].nunique()
    return [y for y in f.years if per_year.get(y, 0) == 2]


def by_scenario(df: pd.DataFrame, index) -> pd.DataFrame:
    """Totaux budget / réalisé côte à côte (colonnes Planned, Realised)."""
    p = df.pivot_table(index=index, columns="scenario", values="amount", aggfunc="sum")
    return p.reindex(columns=[PLANNED, REALISED]).fillna(0).reset_index()


def monthly_pivot(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """Budget et réalisé mensuels sur un calendrier continu ; les mois hors filtre restent vides."""
    p = df.pivot_table(index="date", columns="scenario", values="amount", aggfunc="sum")
    p = p.reindex(columns=[PLANNED, REALISED])
    if p.empty:
        return p.reset_index()
    start = pd.Timestamp(year=min(f.years), month=f.months[0], day=1)
    end = pd.Timestamp(year=max(f.years), month=f.months[1], day=1)
    p = p.reindex(pd.date_range(start, end, freq="MS"))
    keep = p.index.year.isin(f.years) & (p.index.month >= f.months[0]) & (p.index.month <= f.months[1])
    p.loc[~keep] = np.nan
    return p.rename_axis("date").reset_index()


def fc_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Projection annuelle budget / réalisé avec leur plage (colonnes Planned, Planned_low, …)."""
    g = df.groupby(["year", "scenario"])[["amount", "low", "high"]].sum().unstack("scenario")
    out = pd.DataFrame({"year": g.index.to_numpy()})
    for s in (PLANNED, REALISED):
        for col, suffix in (("amount", ""), ("low", "_low"), ("high", "_high")):
            out[s + suffix] = g[(col, s)].to_numpy() if (col, s) in g.columns else 0.0
    return out


def fold_tail(df: pd.DataFrame, n: int, by: str, sums: list[str]) -> pd.DataFrame:
    """Garde les n-1 plus grandes lignes et regroupe le reste dans « Autres »."""
    if len(df) <= n:
        return df
    order = df[by].abs().sort_values(ascending=False).index
    top, rest = df.loc[order[: n - 1]], df.loc[order[n - 1:]]
    other = {c: rest[c].sum() for c in sums}
    other["label"] = f"Autres ({len(rest)} rubriques)"
    return pd.concat([top, pd.DataFrame([other])], ignore_index=True)


# --------------------------------------------------------------------------- aides interface

def show(fig, key: str) -> None:
    st.plotly_chart(fig, theme="streamlit", config=ch.PLOTLY_CONFIG, key=key)


@contextmanager
def card(title: str, caption: str | None = None):
    with st.container(border=True):
        sub = f"<div class='card-sub'>{html.escape(caption)}</div>" if caption else ""
        st.markdown(f"<div class='card-title'>{html.escape(title)}</div>{sub}", unsafe_allow_html=True)
        yield


def tile_row(key: str):
    """Rangée d'indicateurs qui passe à la ligne au lieu de tronquer les valeurs sur écran étroit."""
    return st.container(horizontal=True, gap="small", key=f"kpis_{key}")


def takeaways(lines: list[str], key: str) -> None:
    with st.container(border=True, key=f"insight_{key}"):
        st.markdown("**:material/lightbulb: À retenir**\n\n" + "\n".join(f"- {line}" for line in lines if line))


def section(title: str, caption: str | None = None) -> None:
    st.space("small")
    st.subheader(title, anchor=False)
    if caption:
        st.caption(caption)


def method_note(data: BudgetData, f: Filters) -> None:
    years = data.years
    _, fit = model_checks(data.long, MODEL_VERSION)
    models = "\n".join(f"- **{fc.MODELS[k]}** : {fc.DESCRIPTIONS[k]}." for k in fc.MODELS)
    st.markdown(
        "**Six modèles simples sont calculés pour chaque rubrique, budget et réalisé séparément, et la "
        "sélection automatique retient le meilleur d'entre eux rubrique par rubrique.** "
        f"Modèle affiché : **{fc.MODELS[f.method]}** (à changer dans la barre latérale).\n\n"
        f"{models}\n\n"
        "**Plage** : la zone ombrée couvre l'écart entre les modèles de tendance (linéaire, composée, AR, "
        "forêt aléatoire) ; le naïf saisonnier, sans croissance, sert de référence. **Scénario** : le curseur "
        f"d'ajustement décale toutes les projections de {ch.pct(f.adjust, signed=True, digits=0)}.\n\n"
        "**Test rétrospectif** : chaque modèle apprend sans la dernière année réalisée "
        f"({years[-1]}) puis doit l'estimer ; l'erreur obtenue figure dans la comparaison ci-dessus. "
        "Les modèles apprennent toujours sur toutes les années du classeur ; le filtre des années ne change que "
        "l'affichage, le filtre des mois s'applique aussi aux projections.")
    r2 = fit["r2"].min()
    if np.isfinite(r2) and r2 > 0.9999:
        st.caption(f"Une droite explique les totaux annuels avec un R² de {ch.num(r2, 3)} : l'historique est "
                   "parfaitement linéaire, d'où l'erreur nulle de la tendance linéaire. Sur des données réelles, "
                   "les écarts entre modèles seraient plus marqués.")


def auto_picks_caption(backtest: pd.DataFrame, labels: dict, n_categories: int) -> str:
    """Résumé des modèles retenus par la sélection automatique sur le périmètre affiché."""
    picks = fc.best_models(backtest)
    if picks.empty:
        return "Sélection automatique : historique trop court pour un test, tendance linéaire par défaut."
    if len(picks) == 1:
        row = picks.iloc[0]
        wape = f", écart mensuel moyen {ch.pct(row['wape'])} au test" if np.isfinite(row["wape"]) else ""
        return (f"Sélection automatique pour {md(labels[row['category']])} : "
                f"**{md(fc.MODELS[row['model']])}**{wape}.")
    counts = picks["model"].value_counts()
    parts = [f"{md(fc.MODELS[m])} ({n})" for m, n in counts.items()]
    missing = n_categories - len(picks)
    if missing > 0:
        parts.append(f"tendance linéaire par défaut ({missing})")
    return "Sélection automatique, nombre de rubriques par modèle : " + ", ".join(parts) + "."


def model_comparison(data: BudgetData, f: Filters, block: str, cats, key: str, c: dict) -> None:
    """Projection de l'année suivante selon chaque modèle, et son erreur au test rétrospectif."""
    every = all_forecasts(data.long, MODEL_VERSION)
    year = fc.last_actual_year(data.long) + 1
    sel = every[(every["block"] == block) & (every["scenario"] == f.series) & (every["year"] == year)
                & every["month"].between(*f.months)]
    backtest, _ = model_checks(data.long, MODEL_VERSION)
    scope = backtest[(backtest["block"] == block) & (backtest["scenario"] == f.series)]
    if cats:
        sel, scope = sel[sel["category"].isin(cats)], scope[scope["category"].isin(cats)]
    bt = scope[scope["month"].between(*f.months)]
    factor = np.where(sel["source"] == "forecast", 1 + f.adjust, 1.0)
    totals = (sel.assign(amount=sel["amount"] * factor).groupby("model")["amount"].sum()
              .reindex(list(fc.MODELS)).fillna(0))
    scores = fc.score(bt).set_index("model").reindex(list(fc.MODELS)) if len(bt) else None
    history = scoped(data.long, f, (block,), cats, use_years=False)
    last = year - 1
    last_total = year_total(history, last)
    table = pd.DataFrame({
        "model": list(fc.MODELS),
        "Modèle": [fc.MODELS[m] + (" ✓" if m == f.method else "") for m in fc.MODELS],
        "value": totals.to_numpy(),
    })
    table["growth"] = table["value"] / last_total - 1 if last_total else np.nan
    if scores is not None:
        table["error"] = scores["annual_error"].to_numpy()
        table["mape"] = scores["mape"].to_numpy()
    with card(f"Comparaison des modèles · projection {year}",
              f"{f.series_label} · {f.period} · erreur = test rétrospectif sur {last} (appris sans {last})"):
        show(ch.model_comparison(table, f.unit_for(table["value"]), c, f.method, last_total, last, f.tag),
             f"{key}_models")
        u = f.unit_for(table["value"])
        view = pd.DataFrame({"Modèle": table["Modèle"], f"{year} ({u.label})": u(table["value"])})
        view[f"vs {last} (%)"] = table["growth"] * 100
        if "error" in table:
            view[f"Erreur test {last} (%)"] = table["error"] * 100
            view["Erreur mensuelle moy. (%)"] = table["mape"] * 100
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={
                         f"{year} ({u.label})": st.column_config.NumberColumn(format="localized"),
                         f"vs {last} (%)": st.column_config.NumberColumn(format="%+.1f"),
                         f"Erreur test {last} (%)": st.column_config.NumberColumn(
                             format="%+.1f",
                             help="Total estimé par le modèle / total réel − 1, sur l'année retenue pour le test. Vide pour la "
                                  "sélection automatique : elle est choisie sur ce même test, son erreur serait "
                                  "flatteuse."),
                         "Erreur mensuelle moy. (%)": st.column_config.NumberColumn(
                             format="%.1f", help="Écart absolu moyen mois par mois (MAPE)."),
                     })
        notes = []
        if "error" in table and table["error"].notna().any():
            best = table.loc[table["error"].abs().idxmin()]
            notes.append(f"Meilleur au test sur ce périmètre : **{md(fc.MODELS[best['model']])}** "
                         f"({ch.pct(best['error'], signed=True)} sur le total {last}).")
        n_cats = len(cats) if cats else len(categories_for(data, block))
        notes.append(auto_picks_caption(scope, label_lookup(data), n_cats))
        st.caption("  \n".join(notes))


# --------------------------------------------------------------------------- en-tête

def header(data: BudgetData, f: Filters, fcd: pd.DataFrame) -> None:
    left, right = st.columns([0.62, 0.38], vertical_alignment="bottom")
    with left:
        st.title("Suivi des dépenses", anchor=False)
        unit_txt = f" · montants en {md(f.currency)}" if f.currency else ""
        series_txt = "Budget et réalisé" if compares_both_series() else f.series_label
        st.caption(f"Investissement (CAPEX) et consommation (OPEX) par rubrique · {series_txt} · "
                   f"{f.years_label} · {f.period}{unit_txt}")
    with right:
        with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
            st.badge(f"{len(data.sheets)} feuilles", icon=":material/table_view:", color="gray")
            st.badge(f"Projection jusqu'en {int(fcd['year'].max())}", icon=":material/insights:", color="blue")


def kpi_row(data: BudgetData, f: Filters, fcd: pd.DataFrame) -> None:
    sel = scoped(data.long, f)
    every_year = scoped(data.long, f, use_years=False)
    total = float(sel["amount"].sum())
    monthly = sel.groupby("date")["amount"].sum().sort_index()
    latest = max(f.years)
    prev = latest - 1 if latest - 1 in data.years else None
    first_fc = int(fcd["year"].min())
    fsel = scoped(fcd, f, use_years=False)
    f_total = float(fsel.loc[fsel["year"] == first_fc, "amount"].sum())
    last = last_actual_year(data)
    last_total = year_total(every_year, last)

    with tile_row("header"):
        d = pct(year_total(every_year, latest) / year_total(every_year, prev) - 1) if prev else None
        st.metric(f"Total {f.series_label.lower()}", f.money(total), d, delta_color="off",
                  delta_description=f"{latest} vs {prev}" if prev else None, border=True,
                  chart_data=monthly.tolist(), chart_type="area", height="stretch",
                  help="Somme des feuilles sélectionnées pour les années et mois choisis.")
        for block in BLOCKS:
            b = sel[sel["block"] == block]
            st.metric(BLOCK_TITLE[block], f.money(float(b["amount"].sum())),
                      f"{ch.pct(b['amount'].sum() / total)} du total" if total else None, delta_color="off",
                      delta_arrow="off", border=True, chart_data=b.groupby("date")["amount"].sum().tolist(),
                      chart_type="area", height="stretch")
        st.metric(f"Projection {first_fc}", f.money(f_total), pct(f_total / last_total - 1) if last_total else None,
                  delta_color="off", delta_description=f"vs {last}", border=True, chart_type="line",
                  chart_data=[year_total(every_year, y) for y in data.years] + [f_total], height="stretch",
                  help=f"{fc.MODELS[f.method]} pour {first_fc} ({f.period}), tous blocs confondus. "
                       "Le dernier point est la projection.")


# --------------------------------------------------------------------------- onglets CAPEX / OPEX

def tab_block(data: BudgetData, f: Filters, fcd: pd.DataFrame, c: dict, block: str) -> None:
    labels = label_lookup(data)
    key = block.lower()
    options = [ALL] + categories_for(data, block)
    c1, c2 = st.columns([0.86, 0.14], vertical_alignment="bottom")
    choice = c1.pills("Rubrique", options, default=ALL, key=f"{key}_cat", required=True,
                      format_func=lambda o: "Toutes les rubriques" if o == ALL else labels[o])
    show_fc = c2.toggle("Projection", value=True, key=f"{key}_fc")
    single = choice != ALL
    cats = [choice] if single else None
    title = labels[choice] if single else BLOCK_TITLE[block]

    color = c["realised"] if f.series == REALISED else c["planned"]
    word = f.series_label.lower()
    s = scoped(data.long, f, (block,), cats)
    spend = float(s["amount"].sum())
    if s.empty or spend == 0:
        st.info("Aucune dépense pour cette sélection (rubrique, années ou mois).")
        return
    block_sel = scoped(data.long, f, (block,))
    block_total = float(block_sel["amount"].sum())
    history = scoped(data.long, f, (block,), cats, use_years=False)
    full_year = data.long[(data.long["block"] == block) & (data.long["scenario"] == f.series)
                          & data.long["year"].isin(f.years)]
    if cats:
        full_year = full_year[full_year["category"].isin(cats)]
    fs = scoped(fcd, f, (block,), cats, use_years=False)
    fs = fs if show_fc else fs.iloc[0:0]
    first_fc = int(fcd["year"].min())
    fs1 = fs[fs["year"] == first_fc]
    last = last_actual_year(data)
    last_total = year_total(history, last)
    f_total = float(fs1["amount"].sum())

    by_cat = s.groupby("category")["amount"].sum().sort_values(ascending=False)
    shares = by_cat / spend
    yearly = s.groupby("year")["amount"].sum()
    month_tot = s.groupby("month")["amount"].sum()
    month_share = month_tot / month_tot.sum()
    peak, low = int(month_tot.idxmax()), int(month_tot.idxmin())

    # Saisonnalité : part des dépenses annuelles de chaque rubrique tombant dans chaque mois.
    ph = full_year.groupby(["category", "year", "month"])["amount"].sum().reset_index()
    ph["annual"] = ph.groupby(["category", "year"])["amount"].transform("sum")
    ph = ph[ph["annual"] > 0].assign(share=lambda d: d["amount"] / d["annual"])
    phase = ph.groupby(["category", "month"])["share"].mean().unstack("month").reindex(columns=MONTHS).fillna(0)
    phase = phase.loc[[cat for cat in categories_for(data, block) if cat in phase.index]]

    # À retenir --------------------------------------------------------------------
    first_line = f"{md(title)}, {word} : **{f.money(spend)}** sur {f.years_label} ({f.period})"
    if single:
        first_line += f", soit **{ch.pct(spend / block_total)}** des dépenses {BLOCK_OF[block]}."
    else:
        first_line += (f". La rubrique **{md(labels[by_cat.index[0]])}** en représente à elle seule "
                       f"**{ch.pct(shares.iat[0], digits=0)}**.")
    lines = [first_line]
    if len(yearly) >= 2 and yearly.iat[0] > 0:
        span = yearly.index[-1] - yearly.index[0]
        growth = (yearly.iat[-1] / yearly.iat[0]) ** (1 / span) - 1
        lines.append(f"Croissance de **{pct(growth)} par an** entre {yearly.index[0]} et {yearly.index[-1]} "
                     f"({f.money(yearly.iat[0])} → {f.money(yearly.iat[-1])}).")
    active = (month_tot > 0).sum()
    if len(month_tot) >= 3 and active > 3:
        driver = ""
        if not single:
            avg = s.groupby(["category", "month"])["amount"].sum().unstack("month")
            drop = (avg.mean(axis=1) - avg.get(low, 0)).sort_values(ascending=False)
            if len(drop) and drop.iat[0] > 0:
                driver = f", surtout à cause de la rubrique **{md(labels[drop.index[0]])}**"
        lines.append(f"**{MONTH_NAMES[peak - 1].capitalize()}** est le mois le plus chargé "
                     f"({ch.pct(month_share[peak])} des dépenses) et **{MONTH_NAMES[low - 1]}** le plus calme "
                     f"({ch.pct(month_share[low])}){driver}.")
    lumpy = [cat for cat in phase.index if np.sort(phase.loc[cat].to_numpy())[::-1][:3].sum() >= 0.9
             and (phase.loc[cat] > 0.001).sum() <= 3]
    if lumpy and single:
        months = [MONTH_NAMES[m - 1] for m in MONTHS if phase.loc[lumpy[0], m] > 0.001]
        lines.append(f"Dépense ponctuelle, concentrée en **{', '.join(months)}**.")
    elif lumpy:
        bits = [f"{md(labels[cat])} ({'/'.join(MONTH_ABBR[m - 1] for m in MONTHS if phase.loc[cat, m] > 0.001)})"
                for cat in lumpy[:6]]
        lines.append("Achats ponctuels concentrés sur certains mois : " + ", ".join(bits) + ".")
    if show_fc and f_total:
        change = f", {pct(f_total / last_total - 1)} vs {last}" if last_total else ""
        lines.append(f"Projection **{first_fc}** ({model_phrase(f.method)}) : **{f.money(f_total)}**{change} ; "
                     f"les modèles de tendance donnent "
                     f"{ch.money_range(float(fs1['low'].sum()), float(fs1['high'].sum()), f.unit, f.currency)}.")
    takeaways(lines, key)

    # Indicateurs --------------------------------------------------------------------
    latest = max(f.years)
    prev_total, latest_total = year_total(history, latest - 1), year_total(history, latest)
    monthly = monthly_frame(s, f)
    n_months = s[["year", "month"]].drop_duplicates().shape[0]
    k = tile_row(key)
    k.metric(f"Total {word}", f.money(spend), pct(latest_total / prev_total - 1) if prev_total else None,
             delta_color="off", delta_description=f"{latest} vs {latest - 1}" if prev_total else None, border=True,
             chart_data=monthly["value"].dropna().tolist(), chart_type="area")
    k.metric("Moyenne mensuelle", f.money(spend / max(n_months, 1)), border=True,
             help="Moyenne sur les mois sélectionnés.")
    if single:
        k.metric(f"Part du {block}", ch.pct(spend / block_total), f"sur {f.money(block_total)}", delta_color="off",
                 delta_arrow="off", border=True)
    else:
        top_name = re.sub(r"\s*\([^)]*\)$", "", labels[by_cat.index[0]])
        k.metric("Rubrique principale", ch.pct(shares.iat[0], digits=0), top_name, delta_color="off",
                 delta_arrow="off", border=True, help=f"Part de « {labels[by_cat.index[0]]} » dans la sélection.")
    k.metric("Mois de pointe", MONTH_NAMES[peak - 1].capitalize(), f"{ch.pct(month_share[peak])} des dépenses",
             delta_color="off", delta_arrow="off", border=True)
    if show_fc and len(fs1):
        k.metric(f"Projection {first_fc}", f.money(f_total), pct(f_total / last_total - 1) if last_total else None,
                 delta_color="off", delta_description=f"vs {last}", border=True, chart_type="line",
                 chart_data=[year_total(history, y) for y in data.years] + [f_total])
    else:
        k.metric(f"Projection {first_fc}", "Désactivée", border=True)

    # Évolution mensuelle -------------------------------------------------------------
    cap = f"{f.years_label} · {f.period}"
    if len(fs):
        cap += f" · pointillés = projection, {model_phrase(f.method)} · zone = plage entre modèles"
    with card(f"Dépenses mensuelles · {title}", cap):
        fm = None
        if len(fs):
            fm = (fs.groupby("date")[["amount", "low", "high"]].sum().rename(columns={"amount": "value"})
                  .reset_index())
        vals = monthly["value"].to_numpy()
        if fm is not None:
            vals = np.append(vals, fm["high"].to_numpy())
        show(ch.spend_timeline(monthly, fm, f.unit_for(vals), c, color, f.series_label), f"{key}_timeline")

    # Répartition (ou évolution annuelle) + profil saisonnier ---------------------------
    left, right = st.columns([0.5, 0.5], gap="medium")
    with left:
        if single:
            hist_y = yearly.rename("value").reset_index()
            fy = None
            if len(fs):
                fy = fs.groupby("year")[["amount", "low", "high"]].sum().rename(columns={"amount": "value"}).reset_index()
            vals = np.append(hist_y["value"].to_numpy(), fy["high"].to_numpy() if fy is not None else [])
            with card("Évolution annuelle", f"{f.series_label} · {f.period} · barres claires = projection"):
                show(ch.annual_bars(hist_y, fy, f.unit_for(vals), c, color), f"{key}_annual")
        else:
            mix = pd.DataFrame({"label": [labels[i] for i in by_cat.index], "value": by_cat.to_numpy(),
                                "share": shares.to_numpy()})
            with card("Répartition par rubrique", f"Part des dépenses · {f.years_label} · {f.period}"):
                show(ch.category_bars(mix, f.unit_for(mix["value"]), color), f"{key}_mix")
    with right:
        prof = s.groupby(["year", "month"])["amount"].sum().rename("value").reset_index()
        fprof = fs1.groupby(["year", "month"])["amount"].sum().rename("value").reset_index() if len(fs1) else None
        vals = np.append(prof["value"].to_numpy(), fprof["value"].to_numpy() if fprof is not None else [])
        with card("Profil saisonnier", "Une ligne par année · plus foncé = plus récent · pointillés = projection"):
            show(ch.seasonal_profile(prof, fprof, f.month_list, f.unit_for(vals), c), f"{key}_profile")

    # Calendrier des dépenses (toutes rubriques) -----------------------------------------
    if not single:
        heat = phase.loc[:, f.month_list].copy()
        heat.index = [labels[i] for i in heat.index]
        with card("Calendrier des dépenses",
                  "Part des dépenses annuelles de chaque rubrique tombant dans chaque mois (moyenne des années)"):
            show(ch.phasing_heatmap(heat, c), f"{key}_heat")

        per_year = s.pivot_table(index="category", columns="year", values="amount", aggfunc="sum").fillna(0)
        per_year = per_year.loc[[cat for cat in categories_for(data, block) if cat in per_year.index]]
        u = f.unit_for(per_year.to_numpy())
        trend = s.groupby(["category", "date"])["amount"].sum()
        view = pd.DataFrame({"Rubrique": [labels[i] for i in per_year.index]})
        for y in per_year.columns:
            view[str(y)] = u(per_year[y]).tolist()
        ys = list(per_year.columns)
        if len(ys) >= 2:
            ratio = per_year[ys[-1]] / per_year[ys[0]].replace(0, np.nan)
            view["Croissance / an (%)"] = ((ratio ** (1 / (ys[-1] - ys[0])) - 1) * 100).to_numpy()
        view["Part (%)"] = (per_year.sum(axis=1) / spend * 100).to_numpy()
        view["Tendance mensuelle"] = [trend.loc[cat].tolist() for cat in per_year.index]
        if len(fs1):
            fcat = fs1.groupby("category")["amount"].sum().reindex(per_year.index).fillna(0)
            view[f"{first_fc} P"] = u(fcat).tolist()
        config = {str(y): st.column_config.NumberColumn(format="localized") for y in per_year.columns}
        config |= {
            "Croissance / an (%)": st.column_config.NumberColumn(format="%+.1f"),
            "Part (%)": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
            "Tendance mensuelle": st.column_config.LineChartColumn(y_min=0, width="medium"),
            f"{first_fc} P": st.column_config.NumberColumn(format="localized", help="Projection"),
        }
        with card("Détail par rubrique", f"{f.series_label} par année, en {u.label} · tendance = mois par mois"):
            st.dataframe(view, hide_index=True, width="stretch", column_config=config,
                         height=min(38 + 35 * len(view), 460))

    # Projection -----------------------------------------------------------------------
    description = fc.DESCRIPTIONS[f.method]
    section(f":material/insights: Projection · {title}",
            f"Modèle : {fc.MODELS[f.method]}. {description[0].upper()}{description[1:]}.")
    if fs.empty:
        st.info("La projection est désactivée. Activez l'interrupteur **Projection** en haut de l'onglet.")
        return
    fmon = fs1.groupby("month")["amount"].sum()
    lo, hi = float(fs1["low"].sum()), float(fs1["high"].sum())
    k = tile_row(f"{key}_fc")
    k.metric(f"{f.series_label} {first_fc}", f.money(f_total), pct(f_total / last_total - 1) if last_total else None,
             delta_color="off", delta_description=f"vs {last}", border=True)
    k.metric("Plage entre modèles", ch.money_range(lo, hi, f.unit),
             f"écart {ch.pct((hi - lo) / f_total)}" if f_total else None, delta_color="off", delta_arrow="off",
             border=True,
             help="Plus basse et plus haute des projections des modèles de tendance (linéaire, composée, AR, forêt).")
    k.metric(f"Mois de pointe {first_fc}", MONTH_NAMES[int(fmon.idxmax()) - 1].capitalize(),
             f.money(float(fmon.max())), delta_color="off", delta_arrow="off", border=True)

    model_comparison(data, f, block, cats, key, c)

    if not single:
        cur = fs1.groupby("category")[["amount", "low", "high"]].sum()
        prev_cat = history[history["year"] == last].groupby("category")["amount"].sum()
        fcat = pd.DataFrame({"label": [labels[i] for i in cur.index], "forecast": cur["amount"].to_numpy(),
                             "low": cur["low"].to_numpy(), "high": cur["high"].to_numpy(),
                             "last": prev_cat.reindex(cur.index).fillna(0).to_numpy()})
        fcat = fold_tail(fcat, 12, "forecast", ["forecast", "low", "high", "last"])
        with card(f"Projection {first_fc} par rubrique", f"Barre = projection avec sa plage · trait = réel {last}"):
            show(ch.forecast_by_category(fcat, f.unit_for(fcat[["high", "last"]].to_numpy()), c, f.tag, last,
                                         first_fc), f"{key}_fc_cat")

    if single:  # détail mensuel de la projection pour la rubrique choisie
        rows = fs.pivot_table(index="month", columns="year", values="amount", aggfunc="sum")
        u = f.unit_for(rows.to_numpy())
        table = pd.DataFrame({"Mois": [MONTH_NAMES[m - 1].capitalize() for m in rows.index]})
        last_m = history[history["year"] == last].groupby("month")["amount"].sum().reindex(rows.index).fillna(0)
        table[f"{last} (réel)"] = u(last_m).tolist()
        for y in rows.columns:
            table[f"{y} P"] = u(rows[y]).tolist()
        caption = f"Montants en {u.label} · {f.period}"
    else:
        yearly_fc = fs.pivot_table(index="category", columns="year", values="amount", aggfunc="sum")
        yearly_fc = yearly_fc.loc[[cat for cat in categories_for(data, block) if cat in yearly_fc.index]]
        u = f.unit_for(yearly_fc.to_numpy())
        table = pd.DataFrame({"Rubrique": [labels[i] for i in yearly_fc.index]})
        table[f"{last} (réel)"] = u(history[history["year"] == last].groupby("category")["amount"].sum()
                                    .reindex(yearly_fc.index).fillna(0)).tolist()
        for y in yearly_fc.columns:
            fy = fs[fs["year"] == y].groupby("category")
            table[f"{y} P"] = u(yearly_fc[y]).tolist()
            table[f"{y} bas"] = u(fy["low"].sum().reindex(yearly_fc.index)).tolist()
            table[f"{y} haut"] = u(fy["high"].sum().reindex(yearly_fc.index)).tolist()
        caption = f"Montants en {u.label} · {f.period} · bas/haut = plage entre modèles"
    first_col = table.columns[0]
    totals = {first_col: "Total", **{col: table[col].sum() for col in table.columns[1:]}}
    table = pd.concat([table, pd.DataFrame([totals])], ignore_index=True)
    with card("Tableau de projection", caption):
        st.dataframe(table, hide_index=True, width="stretch", height=min(38 + 35 * len(table), 500),
                     column_config={col: st.column_config.NumberColumn(format="localized")
                                    for col in table.columns[1:]})
    with st.expander("Comment la projection est calculée", icon=":material/help:"):
        method_note(data, f)


# --------------------------------------------------------------------------- onglet écarts budget / réalisé

def gap_backtest(data: BudgetData, f: Filters, blocks, cats) -> tuple[float, float, int] | None:
    """Écart que le modèle choisi aurait projeté pour la dernière année (appris sans elle), et l'écart réel."""
    backtest, _ = model_checks(data.long, MODEL_VERSION)
    if backtest.empty:
        return None
    if f.method == "auto":
        bt = backtest.merge(fc.best_models(backtest)[fc.KEYS + ["model"]], on=fc.KEYS + ["model"])
    else:
        bt = backtest[backtest["model"] == f.method]
    bt = bt[bt["block"].isin(blocks) & bt["month"].between(*f.months)]
    if cats:
        bt = bt[bt["category"].isin(cats)]
    if bt.empty:
        return None
    side = bt.groupby("scenario")[["predicted", "actual"]].sum().reindex([PLANNED, REALISED]).fillna(0)
    predicted = side.at[REALISED, "predicted"] - side.at[PLANNED, "predicted"]
    actual = side.at[REALISED, "actual"] - side.at[PLANNED, "actual"]
    return float(predicted), float(actual), int(bt["year"].iat[0])


def gap_takeaways(f: Filters, title: str, annual: pd.DataFrame, cats_gap: pd.DataFrame, sel: pd.DataFrame,
                  month_gap: pd.DataFrame, fannual: pd.DataFrame | None, single: bool) -> list[str]:
    planned, realised = float(annual[PLANNED].sum()), float(annual[REALISED].sum())
    gap = realised - planned
    ratio = f" ({pct(gap / planned)})" if planned else ""
    side = "au-dessus du budget" if gap > 0 else ("en dessous du budget" if gap < 0 else "exactement au budget")
    lines = [f"{md(title)} : **{f.money(realised)}** réalisés pour **{f.money(planned)}** budgétés, soit "
             f"**{f.money(gap, signed=True)}**{ratio} {side}."]
    if len(annual) >= 2 and (annual[PLANNED] > 0).all():
        r = annual[REALISED] / annual[PLANNED] - 1
        yearly_gap = annual[REALISED] - annual[PLANNED]
        y0, y1 = int(annual["year"].iat[0]), int(annual["year"].iat[-1])
        narrowed = abs(r.iat[-1]) < abs(r.iat[0])
        flat = abs(yearly_gap.mean()) > 0 and np.ptp(yearly_gap) <= 0.01 * abs(yearly_gap.mean())
        if flat:
            amount = f"en montant, il reste stable autour de **{f.money(yearly_gap.mean(), signed=True)}** par an"
            if narrowed and annual[PLANNED].iat[-1] > annual[PLANNED].iat[0]:
                amount += " : le pourcentage baisse seulement parce que le budget augmente"
        else:
            amount = (f"en montant, il passe de **{f.money(yearly_gap.iat[0], signed=True)}** à "
                      f"**{f.money(yearly_gap.iat[-1], signed=True)}**")
        lines.append(f"En pourcentage, l'écart {'se réduit' if narrowed else 'se creuse'} : **{pct(r.iat[0])}** en "
                     f"{y0}, **{pct(r.iat[-1])}** en {y1} ; {amount}.")
    if not single and len(cats_gap) > 1 and gap:
        top = cats_gap.loc[cats_gap["variance"].abs().idxmax()]
        lines.append(f"La rubrique **{md(top['label'])}** explique **{ch.pct(top['variance'] / gap, digits=0)}** "
                     f"de l'écart ({f.money(top['variance'], signed=True)}).")
        per = sel.groupby(["year", "category", "scenario"])["amount"].sum().unstack("scenario")
        per = per.reindex(columns=[PLANNED, REALISED]).fillna(0)
        per = per[per[PLANNED] > 0]
        if len(per) > 1:
            spread = (per[REALISED] / per[PLANNED]).groupby(level="year").agg(lambda s: s.max() - s.min())
            if spread.max() < 5e-4:
                lines.append("Toutes les rubriques s'écartent de leur budget **exactement du même pourcentage** "
                             "chaque année : l'écart est un décalage uniforme, pas quelques lignes qui dérapent.")
    if len(month_gap) and month_gap["gap"].abs().max() > 0:
        biggest = month_gap["gap"].abs().max()
        top = month_gap[month_gap["gap"].abs() >= 0.999 * biggest]
        worst = top.iloc[0]
        month = MONTH_NAMES[int(worst["month"]) - 1]
        when = (f"**{month}**, chaque année" if len(top) > 1 and top["month"].nunique() == 1
                else f"**{month} {int(worst['year'])}**")
        lines.append(f"Plus gros écart mensuel : {when} ({f.money(worst['gap'], signed=True)}).")
    if fannual is not None and len(fannual):
        row = fannual.iloc[0]
        fgap = row[REALISED] - row[PLANNED]
        ratio = f" ({pct(fgap / row[PLANNED])})" if row[PLANNED] else ""
        lines.append(f"Projection **{int(row['year'])}** ({model_phrase(f.method)}) : **{f.money(row[REALISED])}** "
                     f"réalisés pour **{f.money(row[PLANNED])}** budgétés, soit un écart attendu de "
                     f"**{f.money(fgap, signed=True)}**{ratio}.")
    return lines


def tab_gaps(data: BudgetData, f: Filters, fcd: pd.DataFrame, c: dict) -> None:
    labels = label_lookup(data)
    c1, c2 = st.columns([0.86, 0.14], vertical_alignment="bottom")
    scope = c1.segmented_control("Bloc", list(GAP_SCOPES), format_func=GAP_SCOPES.get, default=ALL,
                                 key="gap_block", required=True)
    show_fc = c2.toggle("Projection", value=True, key="gap_fc")
    blocks = BLOCKS if scope == ALL else (scope,)
    choice = ALL
    if scope != ALL:
        choice = st.pills("Rubrique", [ALL] + categories_for(data, scope), default=ALL, key=f"gap_cat_{scope}",
                          required=True, format_func=lambda o: "Toutes les rubriques" if o == ALL else labels[o])
    single = choice != ALL
    cats = [choice] if single else None
    title = labels[choice] if single else ("Toutes les dépenses" if scope == ALL else BLOCK_TITLE[scope])

    comp = comparable_years(data, f)
    if not comp:
        st.info("Sélectionnez au moins une année qui a à la fois une feuille Budget et une feuille Réalisation.")
        return
    sel = scoped(data.long, f, blocks, cats, both=True)
    sel = sel[sel["year"].isin(comp)]
    annual = by_scenario(sel, "year")
    planned, realised = float(annual[PLANNED].sum()), float(annual[REALISED].sum())
    if not planned and not realised:
        st.info("Aucune dépense pour cette sélection (rubrique, années ou mois).")
        return
    gap = realised - planned
    history = scoped(data.long, f, blocks, cats, use_years=False, both=True)
    fsel = scoped(fcd, f, blocks, cats, use_years=False, both=True) if show_fc else fcd.iloc[0:0]
    fannual = fc_by_year(fsel) if len(fsel) else None
    first_fc = int(fcd["year"].min())
    last = last_actual_year(data)

    cats_gap = by_scenario(sel, ["block", "category"])
    cats_gap["variance"] = cats_gap[REALISED] - cats_gap[PLANNED]
    cats_gap["var_pct"] = cats_gap["variance"] / cats_gap[PLANNED].replace(0, np.nan)
    cats_gap["label"] = cats_gap["category"].map(labels)
    month_gap = by_scenario(sel, ["year", "month"])
    month_gap["gap"] = month_gap[REALISED] - month_gap[PLANNED]
    monthly = monthly_pivot(sel, f)

    takeaways(gap_takeaways(f, title, annual, cats_gap, sel, month_gap, fannual, single), "gap")

    # Indicateurs --------------------------------------------------------------------
    def year_side(year, scenario):
        return float(history.loc[(history["year"] == year) & (history["scenario"] == scenario), "amount"].sum())

    latest = max(comp)
    prev = latest - 1 if year_side(latest - 1, PLANNED) and year_side(latest - 1, REALISED) else None
    k = tile_row("gap")
    for label, scenario, total_s in (("Budget", PLANNED, planned), ("Réalisé", REALISED, realised)):
        d = pct(year_side(latest, scenario) / year_side(prev, scenario) - 1) if prev else None
        k.metric(label, f.money(total_s), d, delta_color="off", delta_description=f"{latest} vs {prev}" if prev else None,
                 border=True, chart_data=monthly[scenario].dropna().tolist(), chart_type="area")
    k.metric("Dépassement" if gap >= 0 else "Sous-consommation", f.money(gap, signed=True),
             f"{pct(gap / planned)} du budget" if planned else None, delta_color="inverse", border=True,
             chart_data=(monthly[REALISED] - monthly[PLANNED]).dropna().tolist(), chart_type="line",
             help="Réalisé − budget. Rouge = dépenses au-dessus du budget.")
    rate_delta = None
    if prev and year_side(latest, PLANNED) and year_side(prev, PLANNED):
        r1 = year_side(latest, REALISED) / year_side(latest, PLANNED)
        r0 = year_side(prev, REALISED) / year_side(prev, PLANNED)
        rate_delta = f"{ch.num((r1 - r0) * 100, 1, True)} pt"
    rate = (monthly[REALISED] / monthly[PLANNED].replace(0, np.nan)).dropna()
    k.metric("Taux d'exécution", ch.pct(realised / planned) if planned else "–", rate_delta, delta_color="off",
             delta_description=f"{latest} vs {prev}" if rate_delta else None, border=True,
             chart_data=rate.tolist(), chart_type="line", help="Réalisé en % du budget (100 % = budget respecté).")
    if fannual is not None and len(fannual):
        row = fannual.iloc[0]
        fgap = row[REALISED] - row[PLANNED]
        k.metric(f"Écart attendu {int(row['year'])}", f.money(fgap, signed=True),
                 f"{pct(fgap / row[PLANNED])} du budget" if row[PLANNED] else None, delta_color="inverse",
                 border=True, help=f"Projection ({model_phrase(f.method)}) du réalisé moins celle du budget.")
    else:
        k.metric(f"Écart attendu {first_fc}", "Désactivé", border=True)

    # Années et écart en % ---------------------------------------------------------------
    left, right = st.columns([0.58, 0.42], gap="medium")
    with left:
        vals = annual[[PLANNED, REALISED]].to_numpy()
        if fannual is not None:
            vals = np.append(vals, fannual[[PLANNED, REALISED]].to_numpy())
        with card("Budget et réalisé par année",
                  f"{f.period.capitalize()} · étiquette = réalisé en % du budget · barres claires = projection"):
            show(ch.plan_actual_years(annual, fannual, f.unit_for(vals), c), "gap_years")
    with right:
        with card("Écart en % du budget", "Réalisé − budget, rapporté au budget · 0 % = budget respecté"):
            show(ch.gap_pct_trend(annual, fannual, c), "gap_trend")

    # Mois par mois ---------------------------------------------------------------------
    with card(f"Mois par mois · {title}",
              f"{f.years_label} · en haut : budget et réalisé · en bas : écart mensuel "
              "(rouge ▲ dépassement, vert ▼ sous-consommation)"):
        show(ch.plan_actual_monthly(monthly, f.unit_for(monthly[[PLANNED, REALISED]].to_numpy()), c), "gap_monthly")

    # Écart cumulé + saisonnalité de l'écart -------------------------------------------------
    left, right = st.columns([0.5, 0.5], gap="medium")
    with left:
        fgap_m = None
        if fannual is not None:
            fgap_m = by_scenario(fsel[fsel["year"] == first_fc], ["year", "month"])
            fgap_m["gap"] = fgap_m[REALISED] - fgap_m[PLANNED]
        cum = [month_gap.sort_values(["year", "month"]).groupby("year")["gap"].cumsum().to_numpy()]
        if fgap_m is not None:
            cum.append(fgap_m.sort_values("month")["gap"].cumsum().to_numpy())
        with card("Écart cumulé dans l'année",
                  f"Depuis {MONTH_NAMES[f.months[0] - 1]} · une ligne par année · plus foncé = plus récent · "
                  "pointillés = projection"):
            show(ch.cumulative_gap(month_gap, fgap_m, f.month_list, f.unit_for(np.concatenate(cum)), c), "gap_cumul")
    with right:
        seasonal = month_gap.groupby("month")[[PLANNED, "gap"]].mean().reset_index()
        seasonal["pct"] = seasonal["gap"] / seasonal[PLANNED].replace(0, np.nan)
        with card("Écart moyen par mois de l'année",
                  "Moyenne des années sélectionnées · seul le mois le plus marqué est étiqueté"):
            show(ch.gap_by_month(seasonal, f.unit_for(seasonal["gap"]), c, f.tag), "gap_by_month")

    # Rubriques ------------------------------------------------------------------------
    if not single:
        folded = fold_tail(cats_gap, 12, "variance", [PLANNED, REALISED, "variance"])
        folded["var_pct"] = folded["variance"] / folded[PLANNED].replace(0, np.nan)
        with card("D'où vient l'écart", "Réalisé − budget par rubrique, du plus grand au plus petit · "
                                        "survol = écart en % du budget"):
            show(ch.gap_by_category(folded, f.unit_for(folded["variance"]), c, f.tag), "gap_cats")

        tbl = cats_gap.sort_values("variance", ascending=False)
        yearly = by_scenario(sel, ["category", "year"])
        yearly["p"] = (yearly[REALISED] / yearly[PLANNED].replace(0, np.nan) - 1) * 100
        yearly_pct = yearly.pivot(index="category", columns="year", values="p")
        u = f.unit_for(tbl[[PLANNED, REALISED]].to_numpy())
        view = pd.DataFrame({
            "Rubrique": tbl["label"].to_numpy(), "Bloc": tbl["block"].to_numpy(),
            "Budget": u(tbl[PLANNED]), "Réalisé": u(tbl[REALISED]), "Écart": u(tbl["variance"]),
            "Écart (%)": (tbl["var_pct"] * 100).to_numpy(),
            "Part de l'écart (%)": (tbl["variance"] / gap * 100).to_numpy() if gap else np.nan,
            "Écart par année (%)": [yearly_pct.loc[cat].tolist() for cat in tbl["category"]],
        })
        if scope != ALL:
            view = view.drop(columns="Bloc")
        with card("Tableau des écarts par rubrique",
                  f"Montants en {u.label} · triable · tendance = écart en % du budget, année par année"):
            st.dataframe(view, hide_index=True, width="stretch", height=min(38 + 35 * len(view), 470),
                         column_config={
                             "Budget": st.column_config.NumberColumn(format="localized"),
                             "Réalisé": st.column_config.NumberColumn(format="localized"),
                             "Écart": st.column_config.NumberColumn(format="localized"),
                             "Écart (%)": st.column_config.NumberColumn(format="%+.1f"),
                             "Part de l'écart (%)": st.column_config.ProgressColumn(format="%.1f", min_value=0,
                                                                                    max_value=100),
                             "Écart par année (%)": st.column_config.LineChartColumn(width="small"),
                         })
    else:
        u = f.unit_for(annual[[PLANNED, REALISED]].to_numpy())
        view = pd.DataFrame({
            "Année": annual["year"].astype(str).to_numpy(), "Budget": u(annual[PLANNED]),
            "Réalisé": u(annual[REALISED]), "Écart": u(annual[REALISED] - annual[PLANNED]),
            "Écart (%)": ((annual[REALISED] / annual[PLANNED].replace(0, np.nan) - 1) * 100).to_numpy(),
            "Taux d'exécution (%)": (annual[REALISED] / annual[PLANNED].replace(0, np.nan) * 100).to_numpy(),
        })
        with card("Écarts par année", f"Montants en {u.label} · {f.period}"):
            st.dataframe(view, hide_index=True, width="stretch",
                         column_config={
                             **{col: st.column_config.NumberColumn(format="localized")
                                for col in ("Budget", "Réalisé", "Écart")},
                             "Écart (%)": st.column_config.NumberColumn(format="%+.1f"),
                             "Taux d'exécution (%)": st.column_config.NumberColumn(format="%.1f"),
                         })

    # Projection de l'écart ------------------------------------------------------------------
    section(f":material/insights: Projection de l'écart · {title}",
            f"Budget et réalisé sont projetés séparément ({model_phrase(f.method)}) ; l'écart attendu est leur "
            "différence.")
    if fannual is None or not len(fannual):
        st.info("La projection est désactivée. Activez l'interrupteur **Projection** en haut de l'onglet.")
        return
    years_fc = fannual["year"].astype(int).tolist()
    pick = years_fc[0]
    if len(years_fc) > 1:
        pick = st.segmented_control("Année de projection", years_fc, default=years_fc[0], key="gap_fc_year",
                                    required=True)
    row = fannual[fannual["year"] == pick].iloc[0]
    fgap = row[REALISED] - row[PLANNED]
    p_last, r_last = year_side(last, PLANNED), year_side(last, REALISED)
    k = tile_row("gap_fc")
    k.metric(f"Budget {pick}", f.money(row[PLANNED]), pct(row[PLANNED] / p_last - 1) if p_last else None,
             delta_color="off", delta_description=f"vs {last}", border=True)
    k.metric(f"Réalisé {pick}", f.money(row[REALISED]), pct(row[REALISED] / r_last - 1) if r_last else None,
             delta_color="off", delta_description=f"vs {last}", border=True)
    k.metric("Écart attendu", f.money(fgap, signed=True),
             f"{pct(fgap / row[PLANNED])} du budget" if row[PLANNED] else None, delta_color="inverse", border=True)
    k.metric("Taux d'exécution attendu", ch.pct(row[REALISED] / row[PLANNED]) if row[PLANNED] else "–",
             border=True)
    check = gap_backtest(data, f, blocks, cats)
    if check:
        predicted, actual, test_year = check
        k.metric(f"Test {test_year} : écart projeté", f.money(predicted, signed=True),
                 f"réel {f.money(actual, signed=True)}", delta_color="off", delta_arrow="off", border=True,
                 help=f"Écart que le modèle aurait projeté pour {test_year} en apprenant sans cette année, comparé à "
                      "l'écart réellement constaté."
                      + (" Pour la sélection automatique, le choix des modèles se fait sur ce même test : "
                         "le résultat est flatteur." if f.method == "auto" else ""))

    fy = fsel[fsel["year"] == pick]
    if single:
        g = by_scenario(fy, "month")
        u = f.unit_for(g[[PLANNED, REALISED]].to_numpy())
        table = pd.DataFrame({"Mois": [MONTH_NAMES[m - 1].capitalize() for m in g["month"]]})
    else:
        g = by_scenario(fy, ["block", "category"])
        g = g.assign(label=g["category"].map(labels)).sort_values(REALISED, ascending=False)
        u = f.unit_for(g[[PLANNED, REALISED]].to_numpy())
        table = pd.DataFrame({"Rubrique": g["label"].to_numpy()})
        if scope == ALL:
            table["Bloc"] = g["block"].to_numpy()
    table["Budget"] = u(g[PLANNED])
    table["Réalisé"] = u(g[REALISED])
    table["Écart"] = u(g[REALISED] - g[PLANNED])
    table["Écart (%)"] = ((g[REALISED] / g[PLANNED].replace(0, np.nan) - 1) * 100).to_numpy()
    total_row = {table.columns[0]: "Total", "Budget": table["Budget"].sum(), "Réalisé": table["Réalisé"].sum(),
                 "Écart": table["Écart"].sum()}
    total_row["Écart (%)"] = (total_row["Écart"] / total_row["Budget"] * 100) if total_row["Budget"] else np.nan
    table = pd.concat([table, pd.DataFrame([total_row])], ignore_index=True)
    with card(f"Projection {pick} des écarts", f"Montants en {u.label} · {f.period}"):
        st.dataframe(table, hide_index=True, width="stretch", height=min(38 + 35 * len(table), 500),
                     column_config={
                         **{col: st.column_config.NumberColumn(format="localized")
                            for col in ("Budget", "Réalisé", "Écart")},
                         "Écart (%)": st.column_config.NumberColumn(format="%+.1f"),
                     })
    with st.expander("Comment la projection est calculée", icon=":material/help:"):
        method_note(data, f)


# --------------------------------------------------------------------------- onglet taux de gestion

def rate_components(df: pd.DataFrame, annual_from: pd.DataFrame) -> pd.DataFrame:
    """Composantes des taux ; `annual_from` (tous les mois) fournit le budget de l'année entière."""
    budget = annual_from[annual_from["scenario"] == PLANNED].groupby("year")["amount"].sum()
    return rt.year_components(df, budget)


def rates_definitions(f: Filters) -> None:
    items = "\n".join(f"- **{info['label']}** : {info['formula']}." for info in rt.INDICATOR_INFO.values())
    st.markdown(
        f"{items}\n"
        "- **TCAM** (taux de croissance annuel moyen) : (valeur de fin / valeur de début)^(1 / nombre d'années) − 1.\n"
        "- **Rythme budgété** : budget cumulé depuis janvier / budget de l'année entière ; **rythme linéaire** : "
        "1/12 du budget par mois.\n\n"
        f"Les taux portent sur la période sélectionnée ({f.period}) : avec janv.–sept., le taux de réalisation est "
        "un cumul à fin septembre. Les taux projetés (« P ») sont calculés à partir des projections du budget et "
        f"du réalisé ({model_phrase(f.method)}).\n\n"
        "**Achats consommés** = matières premières, fournitures, pièces de rechange, carburants et lubrifiants ; "
        "le reste de la consommation (sous-traitance, entretien, locations, transports, honoraires, redevances) "
        "forme les services et autres consommations. Regroupement indicatif, proche de la présentation du TCR "
        "(SCF).\n\n"
        "**Non calculables avec ce classeur** : taux d'investissement (investissements / valeur ajoutée), poids "
        "des consommations dans le chiffre d'affaires, taux d'intégration (valeur ajoutée / production), "
        "productivité (valeur ajoutée / effectif). Ils demandent le chiffre d'affaires, la production, la valeur "
        "ajoutée ou les effectifs.")


def tab_rates(data: BudgetData, f: Filters, fcd: pd.DataFrame, c: dict) -> None:
    labels = label_lookup(data)
    comp = comparable_years(data, f)
    if not comp:
        st.info("Sélectionnez au moins une année qui a à la fois une feuille Budget et une feuille Réalisation.")
        return
    c1, c2 = st.columns([0.86, 0.14], vertical_alignment="bottom")
    c1.caption(f"Taux usuels du contrôle de gestion sur la période sélectionnée ({f.period}). Valeurs en %, "
               "années projetées notées « P ». Formules en bas de page.")
    show_fc = c2.toggle("Projection", value=True, key="rates_fc")

    def components(blocks=BLOCKS) -> pd.DataFrame:
        hist = scoped(data.long, f, blocks, use_years=False, both=True)
        out = rate_components(hist, data.long[data.long["block"].isin(blocks)])
        if show_fc:
            fsel = scoped(fcd, f, blocks, use_years=False, both=True)
            if len(fsel):
                out = pd.concat([out, rate_components(fsel, fcd[fcd["block"].isin(blocks)])], ignore_index=True)
        return out

    comps = components()
    ind = rt.indicators(comps)
    fc_years = [int(y) for y in sorted(fcd["year"].unique()) if show_fc and y in ind.columns]
    latest, first = max(comp), min(comp)
    prev = latest - 1 if latest - 1 in ind.columns else None
    partial = f.months != (1, 12)

    def val(key: str, year) -> float:
        if key not in ind.index or year not in ind.columns:
            return np.nan
        return float(ind.at[key, year])

    def delta_pts(key: str) -> tuple[str | None, str | None]:
        if prev is None or not (np.isfinite(val(key, prev)) and np.isfinite(val(key, latest))):
            return None, None
        return f"{ch.num((val(key, latest) - val(key, prev)) * 100, 1, True)} pt", f"{latest} vs {prev}"

    realised_by_year = comps.drop_duplicates("year", keep="last").set_index("year")["R"].sort_index()
    tcam = rt.cagr(realised_by_year.loc[first:latest]) if latest > first else np.nan
    g_real, g_budget = val("growth_real", latest), val("growth_budget", latest)
    s_capex, s_purchases, s_mp = val("share_capex", latest), val("share_purchases", latest), val("share_mp", latest)

    # À retenir ----------------------------------------------------------------------
    head = f"Taux de réalisation global {latest} : **{ch.pct(val('real_total', latest))}**"
    d, _ = delta_pts("real_total")
    if d:
        head += f" ({d} vs {prev})"
    lines = [head + f" ; investissements **{ch.pct(val('real_capex', latest))}**, consommation "
                    f"**{ch.pct(val('real_opex', latest))}**."]
    if len(comp) >= 2:
        a, b = val("real_total", first), val("real_total", latest)
        line = (f"Le taux de réalisation {'recule' if b < a else 'progresse'} : {ch.pct(a)} en {first}, "
                f"{ch.pct(b)} en {latest}")
        if fc_years:
            line += f" ; projection {fc_years[0]} : **{ch.pct(val('real_total', fc_years[0]))}**"
        lines.append(line + ".")
    if np.isfinite(val("over_share", latest)):
        lines.append(f"**{ch.pct(val('over_share', latest), digits=0)}** des rubriques dépassent leur budget "
                     f"en {latest}.")
    if np.isfinite(g_real):
        line = f"Réalisé {latest} : **{pct(g_real)}** sur un an (budget : {pct(g_budget)})"
        if np.isfinite(tcam):
            line += f" ; TCAM {first}–{latest} : **{pct(tcam)}**"
        lines.append(line + ".")
    line = (f"Les investissements représentent **{ch.pct(s_capex)}** des dépenses réalisées ; les achats consommés "
            f"**{ch.pct(s_purchases)}** de la consommation")
    if np.isfinite(s_mp):
        line += f", dont matières premières **{ch.pct(s_mp)}**"
    lines.append(line + ".")
    if partial:
        when = f"À fin {MONTH_NAMES[f.months[1] - 1]} {latest}" if f.months[0] == 1 else f"Sur {f.period} {latest}"
        lines.append(f"{when}, **{ch.pct(val('consumption', latest))}** du budget annuel est consommé.")
    takeaways(lines, "rates")

    # Indicateurs --------------------------------------------------------------------
    k = tile_row("rates")
    for key, label in (("real_total", "Taux de réalisation global"), ("real_capex", "Réalisation des investissements"),
                       ("real_opex", "Réalisation de la consommation")):
        d, desc = delta_pts(key)
        k.metric(label, ch.pct(val(key, latest)), d, delta_color="off", delta_description=desc, border=True,
                 chart_data=(ind.loc[key, comp + fc_years].dropna() * 100).tolist(), chart_type="line",
                 help=rt.INDICATOR_INFO[key]["formula"])
    k.metric(f"Évolution du réalisé {latest}", pct(g_real), f"budget {pct(g_budget)}" if np.isfinite(g_budget) else None,
             delta_color="off", delta_arrow="off", border=True, help=rt.INDICATOR_INFO["growth_real"]["formula"])
    k.metric(f"TCAM {first}–{latest}", pct(tcam), border=True,
             help="Taux de croissance annuel moyen du réalisé sur les années sélectionnées.")
    k.metric("Part des investissements", ch.pct(s_capex), border=True, help=rt.INDICATOR_INFO["share_capex"]["formula"])
    if fc_years:
        fy = fc_years[0]
        diff = val("real_total", fy) - val("real_total", latest)
        k.metric(f"Taux de réalisation {fy} P", ch.pct(val("real_total", fy)),
                 f"{ch.num(diff * 100, 1, True)} pt vs {latest}" if np.isfinite(diff) else None, delta_color="off",
                 border=True, help=f"Projection ({model_phrase(f.method)}) : réalisé projeté / budget projeté.")

    # Tableau de bord des taux -------------------------------------------------------------
    keys = [key for key, *_ in rt.INDICATORS if key in ind.index and (partial or key != "consumption")]
    keys = [key for key in keys if ind.loc[key, comp + fc_years].notna().any()]
    table = pd.DataFrame({
        "Famille": [rt.INDICATOR_INFO[key]["family"] for key in keys],
        "Indicateur": [rt.INDICATOR_INFO[key]["label"] + (f" ({f.period})" if key == "consumption" else "")
                       for key in keys],
    })
    def as_pct(values) -> np.ndarray:  # arrondi : évite les « -0,0 » dus aux erreurs d'arrondi
        return np.round(np.asarray(values, float) * 100, 6) + 0.0

    for y in comp:
        table[str(y)] = as_pct(ind.loc[keys, y])
    for y in fc_years:
        table[f"{y} P"] = as_pct(ind.loc[keys, y])
    if prev is not None:
        table[f"Δ {latest}/{prev} (pt)"] = as_pct(ind.loc[keys, latest] - ind.loc[keys, prev])
    table["Tendance"] = [(ind.loc[key, comp + fc_years].dropna() * 100).tolist() for key in keys]
    table["Formule"] = [rt.INDICATOR_INFO[key]["formula"] for key in keys]
    config = {col: st.column_config.NumberColumn(format="%.1f") for col in table.columns
              if col[:4].isdigit()}
    config |= {"Tendance": st.column_config.LineChartColumn(width="small"),
               "Formule": st.column_config.TextColumn(width="large")}
    if prev is not None:
        config[f"Δ {latest}/{prev} (pt)"] = st.column_config.NumberColumn(format="%+.1f")
    with card("Tableau de bord des taux", f"Valeurs en % · {f.period} · « P » = projection ({model_phrase(f.method)})"):
        st.dataframe(table, hide_index=True, width="stretch", height=38 + 35 * len(table), column_config=config)

    # Analyse d'un taux ----------------------------------------------------------------
    section(":material/query_stats: Analyse d'un taux",
            "Choisissez un taux et un périmètre · années projetées en pointillés ou en plus clair.")
    v1, v2 = st.columns([0.6, 0.4], vertical_alignment="bottom")
    view = v1.pills("Taux", list(RATE_VIEWS), format_func=RATE_VIEWS.get, default="realisation", key="rates_view",
                    required=True)
    scope = v2.segmented_control("Périmètre", list(GAP_SCOPES), format_func=GAP_SCOPES.get, default=ALL,
                                 key="rates_scope", required=True, disabled=view == "structure",
                                 help="Sans objet pour la structure, qui compare les blocs entre eux.")
    blocks = BLOCKS if scope == ALL or view == "structure" else (scope,)
    scope_title = GAP_SCOPES[ALL if view == "structure" else scope]
    hist_scope = scoped(data.long, f, blocks, both=True)
    hist_scope = hist_scope[hist_scope["year"].isin(comp)]
    fsel_scope = scoped(fcd, f, blocks, use_years=False, both=True) if fc_years else fcd.iloc[0:0]

    if view == "realisation":
        left, right = st.columns([0.55, 0.45], gap="medium")
        with left:
            with card("Taux de réalisation par année",
                      f"{f.period.capitalize()} · global, investissement et consommation · 100 % = budget respecté"):
                rows = [("real_total", "Global", c["cat"][0]), ("real_capex", "Investissement (CAPEX)", c["cat"][1]),
                        ("real_opex", "Consommation (OPEX)", c["cat"][2])]
                show(ch.rates_by_year(ind, rows, comp, fc_years, c, ref=100, ref_label="Budget respecté"),
                     "rates_real_years")
        with right:
            quarterly = rt.quarterly_rates(hist_scope)
            fq = rt.quarterly_rates(fsel_scope[fsel_scope["year"] == fc_years[0]]) if fc_years else None
            with card(f"Taux de réalisation trimestriel · {scope_title}",
                      "Une ligne par année · plus foncé = plus récent · pointillés = projection"):
                show(ch.quarterly_rates(quarterly, fq, c), "rates_quarters")
        by_cat = rt.category_rates(hist_scope, latest).assign(label=lambda x: x["category"].map(labels))
        with card(f"Taux de réalisation par rubrique · {scope_title} · {latest}",
                  "Barres à partir de 100 % : ▲ rouge = au-dessus du budget, ▼ vert = en dessous"):
            show(ch.rate_by_category(by_cat, c, latest), "rates_by_cat")

    elif view == "consumption":
        full_scope = data.long[data.long["block"].isin(blocks) & data.long["year"].isin(comp)]
        curves = rt.consumption_curve(full_scope)
        fcur = rt.consumption_curve(fcd[fcd["block"].isin(blocks) & (fcd["year"] == fc_years[0])]) if fc_years else None
        end = f.months[1]
        at_end = curves[(curves["year"] == latest) & (curves["month"] == end)]
        consumed = float(at_end["consumed"].iat[0]) if len(at_end) else np.nan
        pace = float(at_end["budget_pace"].iat[0]) if len(at_end) else np.nan
        month = MONTH_NAMES[end - 1]
        k = tile_row("rates_cons")
        k.metric(f"Consommé à fin {month} {latest}", ch.pct(consumed), border=True,
                 help="Réalisé cumulé depuis janvier / budget de l'année entière.")
        k.metric(f"Rythme budgété à fin {month}", ch.pct(pace), border=True,
                 help="Budget cumulé depuis janvier / budget de l'année entière.")
        k.metric("Écart au rythme budgété", f"{ch.num((consumed - pace) * 100, 1, True)} pt",
                 "en avance sur le budget" if consumed > pace else "en retard sur le budget", delta_color="off",
                 delta_arrow="off", border=True, help="Positif : la consommation va plus vite que prévu au budget.")
        k.metric(f"Rythme linéaire à fin {month}", ch.pct(end / 12), border=True, help="1/12 du budget par mois.")
        if fcur is not None and len(fcur):
            fat = fcur[fcur["month"] == end]
            if len(fat):
                k.metric(f"Consommé à fin {month} {fc_years[0]} P", ch.pct(float(fat["consumed"].iat[0])), border=True,
                         help=f"Projection ({model_phrase(f.method)}).")
        with card(f"Courbe de consommation du budget annuel · {scope_title}",
                  f"Réalisé cumulé depuis janvier / budget de l'année entière · gris : rythme budgété {latest} et "
                  "rythme linéaire · pointillés = projection"):
            show(ch.consumption_curve(curves, fcur, c, latest), "rates_consumption")

    elif view == "growth":
        ind_scope = rt.indicators(components(blocks))
        years_g = [y for y in comp if np.isfinite(ind_scope.at["growth_real", y])]
        left, right = st.columns([0.5, 0.5], gap="medium")
        with left:
            with card(f"Taux d'évolution annuel · {scope_title}",
                      f"{f.period.capitalize()} · réalisé et budget, par rapport à l'année précédente"):
                if years_g or fc_years:
                    rows = [("growth_real", "Réalisé", c["realised"]), ("growth_budget", "Budget", c["planned"])]
                    show(ch.rates_by_year(ind_scope, rows, years_g, fc_years, c, ref=0, ref_label="Stabilité"),
                         "rates_growth")
                else:
                    st.info("Il faut l'année précédente dans le classeur pour calculer un taux d'évolution.")
        with right:
            history = scoped(data.long, f, blocks, use_years=False, both=True)
            if (history["year"] == latest - 1).any():
                growth = rt.category_growth(history, latest).assign(label=lambda x: x["category"].map(labels))
                with card(f"Évolution du réalisé par rubrique · {latest - 1} → {latest}",
                          f"{scope_title} · {f.period}"):
                    show(ch.growth_by_category(growth, c, c["realised"], latest), "rates_growth_cat")
            else:
                st.info(f"Pas de données {latest - 1} pour comparer les rubriques.")

    else:  # structure
        year_labels = [str(y) for y in comp] + [f"{y} P" for y in fc_years]
        fc_labels = {f"{y} P" for y in fc_years}
        spend = pd.DataFrame({"label": year_labels, "capex": [val("share_capex", y) for y in comp + fc_years]})
        spend["opex"] = 1 - spend["capex"]
        consumption = pd.DataFrame({"label": year_labels,
                                    "purchases": [val("share_purchases", y) for y in comp + fc_years]})
        consumption["services"] = 1 - consumption["purchases"]
        left, right = st.columns([0.5, 0.5], gap="medium")
        with left:
            with card("Structure des dépenses réalisées", "Investissement (CAPEX) et consommation (OPEX), en %"):
                show(ch.structure_bars(spend, [("capex", "Investissement (CAPEX)"), ("opex", "Consommation (OPEX)")],
                                       c, fc_labels), "rates_structure_spend")
        with right:
            with card("Structure de la consommation (OPEX)",
                      "Achats consommés et services et autres consommations, en % · proche du TCR (SCF)"):
                show(ch.structure_bars(consumption, [("purchases", "Achats consommés"),
                                                     ("services", "Services et autres consommations")],
                                       c, fc_labels), "rates_structure_cons")
        shares = rt.category_shares(hist_scope)
        if fc_years:
            shares = pd.concat([shares, rt.category_shares(fsel_scope)], ignore_index=True)
        pivot = shares.pivot_table(index=["block", "category"], columns="year", values="share", aggfunc="sum")
        pivot = pivot.reindex(columns=comp + fc_years)
        order = {cat: i for i, cat in enumerate(categories_for(data, CAPEX) + categories_for(data, OPEX))}
        pivot = pivot.reset_index().sort_values(["block", "category"], key=lambda s: s.map(order) if s.name == "category" else s)
        view_tbl = pd.DataFrame({"Rubrique": pivot["category"].map(labels).to_numpy(), "Bloc": pivot["block"].to_numpy()})
        for y in comp:
            view_tbl[str(y)] = (pivot[y] * 100).to_numpy(float)
        for y in fc_years:
            view_tbl[f"{y} P"] = (pivot[y] * 100).to_numpy(float)
        view_tbl["Tendance"] = [[v * 100 for v in row if np.isfinite(v)] for row in pivot[comp + fc_years].to_numpy(float)]
        with card("Part de chaque rubrique dans son bloc", f"Réalisé, en % du bloc · {f.period}"):
            st.dataframe(view_tbl, hide_index=True, width="stretch", height=min(38 + 35 * len(view_tbl), 520),
                         column_config={**{col: st.column_config.NumberColumn(format="%.1f")
                                           for col in view_tbl.columns if col[:4].isdigit()},
                                        "Tendance": st.column_config.LineChartColumn(width="small")})

    with st.expander("Définitions et formules", icon=":material/function:"):
        rates_definitions(f)


# --------------------------------------------------------------------------- main

def main() -> None:
    st.html(CSS)
    file_bytes, name = source_picker()
    if file_bytes is None:
        st.info("Chargez un classeur dans la barre latérale pour commencer.")
        st.stop()
    try:
        data = load(file_bytes, name)
    except Exception as exc:  # afficher le problème de lecture plutôt qu'une trace d'erreur
        st.error(f"Impossible de lire **{md(name)}** : {md(exc)}")
        st.stop()

    f = sidebar(data)
    if not f.years:
        st.warning("Sélectionnez au moins une année dans la barre latérale.")
        st.stop()
    colors = ch.palette(st.context.theme.type)
    fcd = get_forecast(data.long, f.method, f.horizon, f.adjust)

    header(data, f, fcd)
    kpi_row(data, f, fcd)
    st.space("small")
    # Seul l'onglet ouvert est calculé : reruns plus rapides et graphiques toujours à la bonne taille.
    tabs = st.tabs([BLOCK_TABS[CAPEX], BLOCK_TABS[OPEX], GAP_TAB, RATES_TAB], key="main_tab", on_change="rerun")
    renderers = [lambda: tab_block(data, f, fcd, colors, CAPEX), lambda: tab_block(data, f, fcd, colors, OPEX),
                 lambda: tab_gaps(data, f, fcd, colors), lambda: tab_rates(data, f, fcd, colors)]
    for tab, render in zip(tabs, renderers):
        if tab.open:
            with tab:
                render()


main()
