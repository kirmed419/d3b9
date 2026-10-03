"""Taux usuels du contrôle de gestion, calculés à partir du budget et du réalisé.

Tous les taux portent sur la période de mois sélectionnée (par ex. janv.–sept. pour un cumul à fin
septembre). Les années projetées utilisent les projections du budget et du réalisé du modèle choisi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from budget_data import CAPEX, OPEX, PLANNED, REALISED, category_key

# Regroupement indicatif de la consommation, proche de la présentation du TCR (SCF) :
# achats consommés (matières, fournitures, pièces, carburants) / services et autres consommations.
PURCHASE_KEYS = {"mp", "matierespremieres", "fourniture", "fournitures", "carburantsetlubrifiants",
                 "piecesrechange", "piecesderechange"}
RAW_MATERIAL_KEYS = {"mp", "matierespremieres"}

# (clé, famille, libellé, formule)
INDICATORS = [
    ("real_total", "Exécution budgétaire", "Taux de réalisation global", "Réalisé / budget × 100"),
    ("real_capex", "Exécution budgétaire", "Taux de réalisation des investissements (CAPEX)",
     "Réalisé CAPEX / budget CAPEX × 100"),
    ("real_opex", "Exécution budgétaire", "Taux de réalisation de la consommation (OPEX)",
     "Réalisé OPEX / budget OPEX × 100"),
    ("consumption", "Exécution budgétaire", "Taux de consommation du budget annuel",
     "Réalisé de la période / budget de l'année entière × 100"),
    ("over_share", "Exécution budgétaire", "Part des rubriques en dépassement",
     "Rubriques dont le réalisé dépasse le budget / rubriques budgétées × 100"),
    ("growth_real", "Évolution", "Taux d'évolution du réalisé", "(Réalisé N − réalisé N−1) / réalisé N−1 × 100"),
    ("growth_budget", "Évolution", "Taux d'évolution du budget", "(Budget N − budget N−1) / budget N−1 × 100"),
    ("share_capex", "Structure", "Part des investissements dans les dépenses", "Réalisé CAPEX / réalisé total × 100"),
    ("share_purchases", "Structure", "Part des achats consommés dans la consommation",
     "Matières, fournitures, pièces et carburants / réalisé OPEX × 100"),
    ("share_mp", "Structure", "Part des matières premières dans la consommation",
     "Matières premières / réalisé OPEX × 100"),
]
INDICATOR_INFO = {key: {"family": fam, "label": label, "formula": formula} for key, fam, label, formula in INDICATORS}


def _ratio(num, den):
    num = pd.Series(num, dtype=float) if not isinstance(num, pd.Series) else num.astype(float)
    den = pd.Series(den, dtype=float) if not isinstance(den, pd.Series) else den.astype(float)
    return num / den.where(den != 0)


def _sides(df: pd.DataFrame, index) -> pd.DataFrame:
    p = df.groupby(index + ["scenario"])["amount"].sum().unstack("scenario")
    return p.reindex(columns=[PLANNED, REALISED]).fillna(0)


def year_components(df: pd.DataFrame, annual_budget: pd.Series | None = None) -> pd.DataFrame:
    """Totaux utiles aux taux, par année (budget / réalisé par bloc, achats consommés, dépassements).

    `annual_budget` (année -> budget de l'année entière) sert au taux de consommation du budget annuel.
    """
    if df.empty:
        return pd.DataFrame()
    keys = df["category"].map(category_key)
    opex = df["block"] == OPEX
    flags = df.assign(_capex=df["block"] == CAPEX, _opex=opex,
                      _purchase=opex & keys.isin(PURCHASE_KEYS), _mp=opex & keys.isin(RAW_MATERIAL_KEYS))
    rows = []
    for year, g in flags.groupby("year", sort=True):
        def total(scenario, mask=None):
            sel = g["scenario"] == scenario
            if mask is not None:
                sel &= g[mask]
            return float(g.loc[sel, "amount"].sum())

        per_cat = _sides(g, ["category"])
        budgeted = per_cat[per_cat[PLANNED] > 0]
        rows.append({
            "year": int(year), "P": total(PLANNED), "R": total(REALISED),
            "P_CAPEX": total(PLANNED, "_capex"), "R_CAPEX": total(REALISED, "_capex"),
            "P_OPEX": total(PLANNED, "_opex"), "R_OPEX": total(REALISED, "_opex"),
            "R_purchases": total(REALISED, "_purchase"), "R_mp": total(REALISED, "_mp"),
            "has_mp": bool(g["_mp"].any()),
            "n_cats": len(budgeted), "n_over": int((budgeted[REALISED] > budgeted[PLANNED] * (1 + 1e-9)).sum()),
            "P_annual": float(annual_budget.get(year, np.nan)) if annual_budget is not None else np.nan,
        })
    return pd.DataFrame(rows)


def indicators(components: pd.DataFrame) -> pd.DataFrame:
    """Tableau des taux : une ligne par indicateur, une colonne par année (valeurs en fraction, 1 = 100 %)."""
    if components.empty:
        return pd.DataFrame()
    c = components.drop_duplicates("year", keep="last").set_index("year").sort_index()
    prev = c.reindex(c.index - 1)
    prev.index = c.index
    out = pd.DataFrame({
        "real_total": _ratio(c["R"], c["P"]),
        "real_capex": _ratio(c["R_CAPEX"], c["P_CAPEX"]),
        "real_opex": _ratio(c["R_OPEX"], c["P_OPEX"]),
        "consumption": _ratio(c["R"], c["P_annual"]),
        "over_share": _ratio(c["n_over"], c["n_cats"]),
        "growth_real": _ratio(c["R"], prev["R"]) - 1,
        "growth_budget": _ratio(c["P"], prev["P"]) - 1,
        "share_capex": _ratio(c["R_CAPEX"], c["R"]),
        "share_purchases": _ratio(c["R_purchases"], c["R_OPEX"]),
        "share_mp": _ratio(c["R_mp"], c["R_OPEX"]).where(c["has_mp"]),
    }, index=c.index)
    return out.T


def cagr(values: pd.Series) -> float:
    """Taux de croissance annuel moyen (TCAM) entre la première et la dernière année de la série."""
    values = values.dropna()
    if len(values) < 2 or values.iloc[0] <= 0 or values.iloc[-1] <= 0:
        return np.nan
    span = values.index[-1] - values.index[0]
    return (values.iloc[-1] / values.iloc[0]) ** (1 / span) - 1


def quarterly_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Taux de réalisation par trimestre (T1 à T4) et par année."""
    q = _sides(df.assign(quarter=(df["month"] - 1) // 3 + 1), ["year", "quarter"]).reset_index()
    q["rate"] = _ratio(q[REALISED], q[PLANNED])
    return q


def consumption_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Réalisé cumulé depuis janvier et budget cumulé, rapportés au budget de l'année entière."""
    g = _sides(df, ["year", "month"]).reset_index().sort_values(["year", "month"])
    annual = g.groupby("year")[PLANNED].transform("sum")
    g["consumed"] = _ratio(g.groupby("year")[REALISED].cumsum(), annual)
    g["budget_pace"] = _ratio(g.groupby("year")[PLANNED].cumsum(), annual)
    return g[["year", "month", "consumed", "budget_pace"]]


def category_rates(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Taux de réalisation de chaque rubrique pour une année."""
    g = _sides(df[df["year"] == year], ["block", "category"]).reset_index()
    g["rate"] = _ratio(g[REALISED], g[PLANNED])
    return g


def category_growth(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Taux d'évolution du réalisé de chaque rubrique entre l'année précédente et `year`."""
    real = df[df["scenario"] == REALISED]
    cur = real[real["year"] == year].groupby(["block", "category"])["amount"].sum()
    prev = real[real["year"] == year - 1].groupby(["block", "category"])["amount"].sum()
    out = pd.DataFrame({"R": cur, "R_prev": prev.reindex(cur.index)}).reset_index()
    out["growth"] = _ratio(out["R"], out["R_prev"]) - 1
    return out


def category_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Part du réalisé de chaque rubrique dans son bloc, par année."""
    real = df[df["scenario"] == REALISED].groupby(["block", "category", "year"])["amount"].sum().reset_index()
    real["share"] = _ratio(real["amount"], real.groupby(["block", "year"])["amount"].transform("sum"))
    return real
