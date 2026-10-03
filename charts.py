"""Graphiques Plotly partageant un même langage visuel sobre.

Rôles des couleurs (validés avec les contrôles de palette dataviz) :
- réalisé = bleu, budget = gris neutre (bleu/gris : ΔE daltonisme 15,9, vision normale 17,8, contraste
  >= 3:1 en clair comme en sombre) ;
- dépassement = rouge, sous-consommation = vert (couleurs de statut, toujours avec ▲ / ▼ ou un signe) ;
- les montants (carte de chaleur) utilisent une seule rampe bleue, inversée en mode sombre ;
- les années utilisent une rampe ordinale bleue, l'année la plus récente étant la plus marquée.
Les projections sont toujours en pointillés ou en remplissage clair, dans une zone « Projection ».
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from budget_data import MONTH_ABBR

FONT = "Inter, 'Source Sans 3', 'Source Sans Pro', system-ui, -apple-system, 'Segoe UI', sans-serif"

COLORS = {
    "light": {
        "realised": "#2a78d6", "planned": "#898781", "ink": "#0b0b0b", "muted": "#898781",
        "over": "#d03b3b", "under": "#0ca30c",  # statuts fixes, toujours accompagnés de ▲ / ▼
        # 3 premières teintes catégorielles : validées « toutes paires » dans les deux modes (palette.md).
        "cat": ["#2a78d6", "#eb6834", "#1baf7a"],
        "zone": "rgba(137,135,129,0.10)",
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        # Rampes ordinales des années (ancienne -> récente) ; chaque jeu passe les contrôles (ΔL >= 0,06).
        "years": {1: ["#2a78d6"], 2: ["#86b6ef", "#1c5cab"], 3: ["#86b6ef", "#256abf", "#0d366b"],
                  4: ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
                  5: ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]},
    },
    "dark": {
        "realised": "#3987e5", "planned": "#898781", "ink": "#ffffff", "muted": "#898781",
        "over": "#d03b3b", "under": "#0ca30c",
        "cat": ["#3987e5", "#d95926", "#199e70"],
        "zone": "rgba(195,194,183,0.08)",
        "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "years": {1: ["#3987e5"], 2: ["#256abf", "#86b6ef"], 3: ["#184f95", "#3987e5", "#b7d3f6"],
                  4: ["#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"],
                  5: ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]},
    },
}
MAX_PROFILE_YEARS = 5

PLOTLY_CONFIG = {
    "displaylogo": False,
    "locale": "fr",
    "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
                               "autoScale2d", "resetScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

_SCALES = {"Md": 1e9, "M": 1e6, "k": 1e3}


def palette(mode: str | None) -> dict:
    return COLORS["dark" if mode == "dark" else "light"]


# --------------------------------------------------------------------------- formatage (français)

def num(value: float, digits: int = 2, signed: bool = False) -> str:
    """1234567.8 -> '1 234 567,80' (espace fine comme séparateur de milliers, virgule décimale)."""
    value = round(float(value), digits) + 0.0  # évite « -0,0 »
    text = f"{value:+,.{digits}f}" if signed and value else f"{value:,.{digits}f}"
    return text.replace(",", " ").replace(".", ",")


@dataclass(frozen=True)
class Unit:
    factor: float = 1e9
    suffix: str = "Md"
    currency: str = ""

    @property
    def label(self) -> str:
        return " ".join(p for p in (self.suffix, self.currency) if p)

    def __call__(self, values) -> np.ndarray:
        return np.asarray(values, dtype=float) / self.factor


def choose_unit(choice: str, magnitude: float, currency: str = "") -> Unit:
    if choice in _SCALES:
        return Unit(_SCALES[choice], choice, currency)
    for suffix, factor in _SCALES.items():
        if abs(magnitude) >= factor:
            return Unit(factor, suffix, currency)
    return Unit(1.0, "", currency)


def money(value: float, choice: str = "Auto", currency: str = "", signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    unit = choose_unit(choice, value, currency)
    scaled = value / unit.factor
    return f"{num(scaled, 2 if abs(scaled) < 100 else 1, signed)} {unit.label}".strip()


def money_range(low: float, high: float, choice: str = "Auto", currency: str = "") -> str:
    unit = choose_unit(choice, max(abs(low), abs(high)), currency)
    digits = 2 if abs(high / unit.factor) < 100 else 1
    return f"{num(low / unit.factor, digits)}–{num(high / unit.factor, digits)} {unit.label}".strip()


def pct(value: float, signed: bool = False, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    return f"{num(value * 100, digits, signed)} %"


def short_labels(labels, limit: int = 26) -> list[str]:
    """Raccourcit les libellés d'axe trop longs (le libellé complet reste au survol) en les gardant uniques."""
    out, seen = [], set()
    for label in map(str, labels):
        text = label if len(label) <= limit else label[: limit - 1].rstrip() + "…"
        while text in seen:
            text += "​"
        seen.add(text)
        out.append(text)
    return out


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _base(fig: go.Figure, height: int, *, legend: bool = True, hovermode: str | bool = "x unified",
          top: int | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=16, t=top if top is not None else (34 if legend else 14), b=4),
        font=dict(family=FONT, size=12), separators=", ",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode=hovermode, hoverlabel=dict(font=dict(family=FONT, size=12), namelength=-1),
        showlegend=legend,
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom", title_text="",
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
        barcornerradius=4, dragmode=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="", automargin=True,
                     tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridwidth=1, zeroline=False, showline=False, ticks="", automargin=True,
                     tickfont=dict(size=11), title_font=dict(size=11))
    return fig


def _forecast_zone(fig, x0, x1, c):
    fig.add_vrect(x0=x0, x1=x1, fillcolor=c["zone"], line_width=0, layer="below",
                  annotation_text="Projection", annotation_position="top left",
                  annotation_font=dict(size=11, color=c["muted"]))


def _month_ticks(fig, dates, **kw):
    """Graduations trimestrielles en français (Plotly n'embarque pas les noms de mois français)."""
    dates = pd.Series(pd.to_datetime(dates)).dropna()
    ticks = pd.date_range(dates.min(), dates.max(), freq="3MS") if len(dates) else []
    fig.update_xaxes(tickvals=list(ticks), ticktext=[f"{MONTH_ABBR[t.month - 1]}<br>{t.year}" for t in ticks], **kw)


def _signed_colors(values, c) -> list[str]:
    return [c["over"] if v > 0 else c["under"] for v in np.nan_to_num(np.asarray(values, float))]


# --------------------------------------------------------------------------- graphiques

def spend_timeline(hist: pd.DataFrame, fc: pd.DataFrame | None, unit: Unit, c: dict, color: str,
                   name: str) -> go.Figure:
    """Dépenses mensuelles en aire légère, prolongées par une projection en pointillés avec sa plage."""
    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=unit(hist["value"]), name=name, mode="lines", connectgaps=False,
                    line=dict(color=color, width=2), fill="tozeroy", fillcolor=_rgba(color, 0.10),
                    hovertemplate=f"{name} : %{{y:,.2f}} {unit.label}<extra></extra>")
    if fc is not None and len(fc):
        fig.add_scatter(x=fc["date"], y=unit(fc["high"]), mode="lines", line=dict(width=0), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=fc["date"], y=unit(fc["low"]), mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor=_rgba(color, 0.18), name="Plage entre modèles",
                        customdata=unit(fc["high"]),
                        hovertemplate=f"Plage : %{{y:,.2f}}–%{{customdata:,.2f}} {unit.label}<extra></extra>")
        fx, fy = list(fc["date"]), list(unit(fc["value"]))
        last = hist.dropna(subset=["value"]).tail(1)
        if len(last) and (fc["date"].min() - last["date"].iat[0]).days <= 31:
            fx, fy = [last["date"].iat[0]] + fx, [float(unit(last["value"])[0])] + fy
        fig.add_scatter(x=fx, y=fy, name="Projection", mode="lines", line=dict(color=color, width=2, dash="dash"),
                        hovertemplate=f"Projection : %{{y:,.2f}} {unit.label}<extra></extra>")
        _forecast_zone(fig, fc["date"].min() - pd.Timedelta(days=16), fc["date"].max() + pd.Timedelta(days=16), c)
    _base(fig, 360)
    fig.update_yaxes(title_text=unit.label, rangemode="tozero")
    months = pd.concat([hist["date"], fc["date"] if fc is not None else pd.Series(dtype="datetime64[ns]")])
    ticks = pd.date_range(months.min(), months.max(), freq="3MS") if len(months) else []
    fig.update_xaxes(tickvals=list(ticks),
                     ticktext=[f"{MONTH_ABBR[t.month - 1]}<br>{t.year}" for t in ticks])
    return fig


def annual_bars(hist: pd.DataFrame, fc: pd.DataFrame | None, unit: Unit, c: dict, color: str) -> go.Figure:
    """Total par année (une couleur) ; les années de projection sont plus claires, avec leur plage."""
    fig = go.Figure()
    xs = [str(y) for y in hist["year"]]
    growth = hist["value"].pct_change()
    fig.add_bar(x=xs, y=unit(hist["value"]), name="Réel", marker_color=color,
                text=[money(v, "Auto") for v in hist["value"]], textposition="outside", cliponaxis=False,
                constraintext="none", textfont=dict(size=11), customdata=growth,
                hovertemplate=f"%{{x}} : %{{y:,.2f}} {unit.label}<extra></extra>")
    top = float(np.nanmax(unit(hist["value"]))) if len(hist) else 0.0
    if fc is not None and len(fc):
        xf = [f"{y} P" for y in fc["year"]]
        fig.add_bar(x=xf, y=unit(fc["value"]), name="Projection", marker_color=_rgba(color, 0.42),
                    error_y=dict(type="data", symmetric=False, array=unit(fc["high"] - fc["value"]),
                                 arrayminus=unit(fc["value"] - fc["low"]), thickness=1.2, width=4,
                                 color=c["muted"]),
                    text=[money(v, "Auto") for v in fc["value"]], textposition="outside", cliponaxis=False,
                    constraintext="none", textfont=dict(size=11),
                    customdata=np.stack([unit(fc["low"]), unit(fc["high"])], axis=-1),
                    hovertemplate=(f"%{{x}} : %{{y:,.2f}} {unit.label}"
                                   " (plage %{customdata[0]:,.2f}–%{customdata[1]:,.2f})<extra></extra>"))
        top = max(top, float(np.nanmax(unit(fc["high"]))))
        _forecast_zone(fig, len(xs) - 0.5, len(xs) + len(xf) - 0.5, c)
    _base(fig, 360, legend=False, hovermode="closest")
    fig.update_layout(bargap=0.5)
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text=unit.label, range=[0, top * 1.2 if top else 1])
    return fig


def category_bars(df: pd.DataFrame, unit: Unit, color: str) -> go.Figure:
    """Classement des rubriques (une série, une couleur), part du total au bout de la barre."""
    d = df.sort_values("value")
    x = unit(d["value"])
    share_txt = [pct(s, digits=1 if s < 0.1 else 0) for s in d["share"]]
    fig = go.Figure(go.Bar(
        y=short_labels(d["label"], 24), x=x, orientation="h", marker_color=color,
        customdata=list(zip(d["label"].astype(str), d["share"].astype(float))),
        text=share_txt, textposition="outside", cliponaxis=False, constraintext="none", textfont=dict(size=11),
        hovertemplate=f"%{{customdata[0]}}<br>%{{x:,.3f}} {unit.label} · %{{customdata[1]:.1%}} du total<extra></extra>",
    ))
    _base(fig, max(240, 30 * len(d) + 50), legend=False, hovermode="closest")
    fig.update_layout(bargap=0.38)
    fig.update_xaxes(range=[0, float(np.nanmax(x)) * 1.25 if len(x) else 1], showgrid=True,
                     title_text=unit.label)
    fig.update_yaxes(showgrid=False)
    return fig


def seasonal_profile(hist: pd.DataFrame, fc: pd.DataFrame | None, months: list[int], unit: Unit,
                     c: dict) -> go.Figure:
    """Une ligne par année sur les mois ; années récentes plus marquées, projection en pointillés (5 lignes max)."""
    fyears = sorted(fc["year"].unique())[:1] if fc is not None and len(fc) else []
    years = sorted(hist["year"].unique())[-(MAX_PROFILE_YEARS - len(fyears)):]
    ramp = c["years"][max(1, len(years) + len(fyears))]
    x = [MONTH_ABBR[m - 1] for m in months]
    fig = go.Figure()
    for i, year in enumerate(years + fyears):
        is_fc = year in fyears
        src = fc if is_fc else hist
        y = src[src["year"] == year].set_index("month")["value"].reindex(months)
        name = f"{year} (projection)" if is_fc else str(year)
        fig.add_scatter(x=x, y=unit(y), name=name, mode="lines+markers",
                        line=dict(color=ramp[i], width=2, dash="dash" if is_fc else "solid"),
                        marker=dict(size=7, color=ramp[i], symbol="circle-open" if is_fc else "circle",
                                    line=dict(width=2, color=ramp[i])),
                        hovertemplate=f"{name} : %{{y:,.2f}} {unit.label}<extra></extra>")
    _base(fig, 360)
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text=unit.label, rangemode="tozero")
    return fig


def phasing_heatmap(share: pd.DataFrame, c: dict) -> go.Figure:
    """Grille rubrique x mois : part des dépenses annuelles de chaque rubrique tombant dans chaque mois."""
    z = share.to_numpy(float) * 100
    zmax = float(np.nanmax(z)) if z.size else 1.0
    steps = c["seq"]
    scale = [[i / (len(steps) - 1), col] for i, col in enumerate(steps)]
    text = [[f"{v:.0f} %" if v >= 0.5 else "" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=[MONTH_ABBR[m - 1] for m in share.columns], y=list(share.index), colorscale=scale,
        zmin=0, zmax=max(zmax, 1.0), xgap=2, ygap=2, text=text, texttemplate="%{text}",
        textfont=dict(size=10), hovertemplate="%{y}<br>%{x} : %{z:.1f} % de l'année<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0, ticksuffix=" %", len=0.85, tickfont=dict(size=10)),
    ))
    _base(fig, max(260, 30 * len(share) + 60), legend=False, hovermode="closest", top=8)
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def model_comparison(table: pd.DataFrame, unit: Unit, c: dict, selected: str, last_total: float, last_year: int,
                     fmt) -> go.Figure:
    """Projection de l'année suivante par modèle : le modèle choisi en bleu, les autres en gris (mise en avant)."""
    d = table.iloc[::-1]
    colors = [c["realised"] if m == selected else _rgba(c["planned"], 0.55) for m in d["model"]]
    text = [f"{fmt(v)} ({pct(g, signed=True)})" if np.isfinite(g) else fmt(v)
            for v, g in zip(d["value"], d["growth"])]
    fig = go.Figure(go.Bar(
        y=d["Modèle"], x=unit(d["value"]), orientation="h", marker_color=colors, text=text,
        textposition="outside", cliponaxis=False, constraintext="none", textfont=dict(size=11),
        hovertemplate=f"%{{y}} : %{{x:,.2f}} {unit.label}<extra></extra>",
    ))
    if last_total:
        fig.add_vline(x=float(unit(last_total)), line=dict(color=c["ink"], width=1.5),
                      annotation_text=f"{last_year} réel", annotation_position="top",
                      annotation_font=dict(size=11, color=c["muted"]))
    _base(fig, 300, legend=False, hovermode="closest", top=26)
    fig.update_layout(bargap=0.38)
    top = float(np.nanmax(unit(d["value"]))) if len(d) else 1
    fig.update_xaxes(range=[0, top * 1.45], showgrid=True, title_text=unit.label)
    fig.update_yaxes(showgrid=False)
    return fig


def forecast_by_category(df: pd.DataFrame, unit: Unit, c: dict, fmt, last_year: int, year: int) -> go.Figure:
    """Projection par rubrique (barre + plage des méthodes) comparée au réel de l'an dernier (trait)."""
    d = df.sort_values("forecast")
    growth = (d["forecast"] / d["last"].replace(0, np.nan) - 1).astype(float)
    names = short_labels(d["label"], 30)
    full = d["label"].astype(str)
    fig = go.Figure()
    fig.add_bar(
        y=names, x=unit(d["forecast"]), orientation="h", name=f"Projection {year}",
        marker_color=_rgba(c["realised"], 0.55),
        error_x=dict(type="data", symmetric=False, array=unit(d["high"] - d["forecast"]),
                     arrayminus=unit(d["forecast"] - d["low"]), thickness=1.2, width=3, color=c["muted"]),
        text=[f"{fmt(v)} ({pct(g, signed=True)})" if np.isfinite(g) else fmt(v)
              for v, g in zip(d["forecast"], growth)],
        textposition="outside", cliponaxis=False, constraintext="none", textfont=dict(size=11),
        customdata=list(zip(full, growth)),
        hovertemplate=(f"%{{customdata[0]}}<br>Projection {year} : %{{x:,.3f}} {unit.label}"
                       " (%{customdata[1]:+.1%})<extra></extra>"),
    )
    fig.add_scatter(
        y=names, x=unit(d["last"]), mode="markers", name=f"{last_year} (réel)", customdata=full,
        marker=dict(symbol="line-ns", size=18, line=dict(width=3, color=c["ink"])),
        hovertemplate=f"%{{customdata}}<br>{last_year} : %{{x:,.3f}} {unit.label}<extra></extra>",
    )
    _base(fig, max(260, 30 * len(d) + 70), hovermode="closest")
    fig.update_layout(bargap=0.38)
    top = float(np.nanmax(unit(d[["high", "last"]].to_numpy()))) if len(d) else 1
    fig.update_xaxes(range=[0, top * 1.45], showgrid=True, title_text=unit.label)
    fig.update_yaxes(showgrid=False)
    return fig


# --------------------------------------------------------------------------- écarts budget / réalisé

def _pct_label(value: float) -> str:
    return f"{num(value, 1, True)} %" if np.isfinite(value) else ""


def plan_actual_years(hist: pd.DataFrame, fc: pd.DataFrame | None, unit: Unit, c: dict) -> go.Figure:
    """Colonnes groupées par année : budget (gris) et réalisé (bleu) ; années projetées en plus clair.

    L'étiquette au-dessus du réalisé donne le taux d'exécution (réalisé en % du budget).
    """
    fig = go.Figure()
    xs = [str(y) for y in hist["year"]]
    rate = (hist["Realised"] / hist["Planned"].replace(0, np.nan)).astype(float)
    fig.add_bar(x=xs, y=unit(hist["Planned"]), name="Budget", marker_color=c["planned"], offsetgroup="p",
                legendgroup="p", hovertemplate=f"Budget : %{{y:,.2f}} {unit.label}<extra></extra>")
    fig.add_bar(x=xs, y=unit(hist["Realised"]), name="Réalisé", marker_color=c["realised"], offsetgroup="r",
                legendgroup="r", customdata=rate, text=[pct(r) for r in rate], textposition="outside",
                cliponaxis=False, constraintext="none", textfont=dict(size=11),
                hovertemplate=f"Réalisé : %{{y:,.2f}} {unit.label} · %{{customdata:.1%}} du budget<extra></extra>")
    top = float(np.nanmax(unit(hist[["Planned", "Realised"]].to_numpy()))) if len(hist) else 0.0
    if fc is not None and len(fc):
        xf = [f"{y} P" for y in fc["year"]]
        frate = (fc["Realised"] / fc["Planned"].replace(0, np.nan)).astype(float)
        for name, col, color, group, labels in (("Budget", "Planned", c["planned"], "p", None),
                                                ("Réalisé", "Realised", c["realised"], "r", [pct(r) for r in frate])):
            fig.add_bar(
                x=xf, y=unit(fc[col]), name=f"{name} (projection)", marker_color=_rgba(color, 0.42),
                offsetgroup=group, legendgroup=group, showlegend=False, text=labels, textposition="outside",
                cliponaxis=False, constraintext="none", textfont=dict(size=11),
                customdata=np.stack([unit(fc[f"{col}_low"]), unit(fc[f"{col}_high"])], axis=-1),
                hovertemplate=(f"{name} projeté : %{{y:,.2f}} {unit.label}"
                               " (plage %{customdata[0]:,.2f}–%{customdata[1]:,.2f})<extra></extra>"))
        top = max(top, float(np.nanmax(unit(fc[["Planned", "Realised"]].to_numpy()))))
        _forecast_zone(fig, len(xs) - 0.5, len(xs) + len(xf) - 0.5, c)
    _base(fig, 340)
    fig.update_layout(barmode="group", bargap=0.42, bargroupgap=0.14)
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text=unit.label, range=[0, top * 1.18 if top else 1])
    return fig


def gap_pct_trend(hist: pd.DataFrame, fc: pd.DataFrame | None, c: dict) -> go.Figure:
    """Écart réalisé − budget en % du budget, par année (0 % = conforme au budget)."""
    fig = go.Figure()
    xs = [str(y) for y in hist["year"]]
    ys = ((hist["Realised"] / hist["Planned"].replace(0, np.nan) - 1) * 100).to_numpy(float)
    fig.add_scatter(x=xs, y=ys, mode="lines+markers+text", name="Réel", text=[_pct_label(v) for v in ys],
                    textposition="top center", textfont=dict(size=11), cliponaxis=False,
                    line=dict(color=c["realised"], width=2), marker=dict(size=9, color=c["realised"]),
                    hovertemplate="%{x} : %{y:+.2f} % du budget<extra></extra>")
    values = list(ys)
    if fc is not None and len(fc) and len(xs):
        xf = [f"{y} P" for y in fc["year"]]
        yf = ((fc["Realised"] / fc["Planned"].replace(0, np.nan) - 1) * 100).to_numpy(float)
        values += list(yf)
        fig.add_scatter(x=[xs[-1]] + xf, y=[ys[-1]] + list(yf), mode="lines+markers+text", name="Projection",
                        text=[""] + [_pct_label(v) for v in yf], textposition="top center",
                        textfont=dict(size=11), cliponaxis=False,
                        line=dict(color=c["realised"], width=2, dash="dash"),
                        marker=dict(size=9, symbol="circle-open", line=dict(width=2), color=c["realised"]),
                        hovertemplate="%{x} : %{y:+.2f} % du budget<extra></extra>")
        _forecast_zone(fig, len(xs) - 0.5, len(xs) + len(xf) - 0.5, c)
    fig.add_hline(y=0, line=dict(color=c["muted"], width=1), annotation_text="Conforme au budget",
                  annotation_position="top left", annotation_font=dict(size=11, color=c["muted"]))
    _base(fig, 340, legend=False, hovermode="closest")
    finite = [v for v in values if np.isfinite(v)]
    lo, hi = min([0.0] + finite), max([0.0] + finite)
    pad = (hi - lo) * 0.25 or 1
    fig.update_xaxes(type="category")
    fig.update_yaxes(ticksuffix=" %", range=[lo - (pad if lo < 0 else 0), hi + pad], title_text="Écart / budget")
    return fig


def plan_actual_monthly(m: pd.DataFrame, unit: Unit, c: dict) -> go.Figure:
    """Budget et réalisé mois par mois (en haut) et l'écart mensuel (en bas), sur le même axe du temps."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.07)
    fig.add_scatter(x=m["date"], y=unit(m["Planned"]), name="Budget", mode="lines", connectgaps=False,
                    line=dict(color=c["planned"], width=2),
                    hovertemplate=f"Budget : %{{y:,.2f}} {unit.label}<extra></extra>", row=1, col=1)
    fig.add_scatter(x=m["date"], y=unit(m["Realised"]), name="Réalisé", mode="lines", connectgaps=False,
                    line=dict(color=c["realised"], width=2.5),
                    hovertemplate=f"Réalisé : %{{y:,.2f}} {unit.label}<extra></extra>", row=1, col=1)
    gap = (m["Realised"] - m["Planned"]).to_numpy(float)
    fig.add_bar(x=m["date"], y=unit(gap), name="Écart", showlegend=False, marker_color=_signed_colors(gap, c),
                hovertemplate=f"Écart : %{{y:+,.2f}} {unit.label}<extra></extra>", row=2, col=1)
    _base(fig, 440)
    fig.update_layout(bargap=0.5)
    fig.update_yaxes(title_text=unit.label, rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Écart", zeroline=True, zerolinewidth=1, row=2, col=1)
    _month_ticks(fig, m["date"], row=2, col=1)
    return fig


def cumulative_gap(hist: pd.DataFrame, fc: pd.DataFrame | None, months: list[int], unit: Unit,
                   c: dict) -> go.Figure:
    """Écart cumulé depuis le premier mois affiché, une ligne par année (récente = plus marquée)."""
    fyears = sorted(fc["year"].unique())[:1] if fc is not None and len(fc) else []
    years = sorted(hist["year"].unique())[-(MAX_PROFILE_YEARS - len(fyears)):]
    ramp = c["years"][max(1, len(years) + len(fyears))]
    x = [MONTH_ABBR[m - 1] for m in months]
    fig = go.Figure()
    for i, year in enumerate(years + fyears):
        is_fc = year in fyears
        src = fc if is_fc else hist
        y = src[src["year"] == year].set_index("month")["gap"].reindex(months).fillna(0).cumsum()
        name = f"{year} (projection)" if is_fc else str(year)
        fig.add_scatter(x=x, y=unit(y), name=name, mode="lines+markers",
                        line=dict(color=ramp[i], width=2, dash="dash" if is_fc else "solid"),
                        marker=dict(size=7, color=ramp[i], symbol="circle-open" if is_fc else "circle",
                                    line=dict(width=2, color=ramp[i])),
                        hovertemplate=f"{name} : %{{y:+,.2f}} {unit.label} cumulés<extra></extra>")
    fig.add_hline(y=0, line=dict(color=c["muted"], width=1))
    _base(fig, 340)
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text=f"Écart cumulé ({unit.label})")
    return fig


def gap_by_month(df: pd.DataFrame, unit: Unit, c: dict, fmt) -> go.Figure:
    """Écart moyen par mois de l'année ; seul le mois le plus marqué porte une étiquette."""
    x = [MONTH_ABBR[m - 1] for m in df["month"]]
    gaps = df["gap"].to_numpy(float)
    peak = int(np.nanargmax(np.abs(gaps))) if len(gaps) else -1
    text = [f"{'▲' if v > 0 else '▼'} {fmt(abs(v))}" if i == peak else "" for i, v in enumerate(gaps)]
    fig = go.Figure(go.Bar(
        x=x, y=unit(gaps), marker_color=_signed_colors(gaps, c), customdata=df["pct"].astype(float), text=text,
        textposition="outside", cliponaxis=False, constraintext="none", textfont=dict(size=11),
        hovertemplate=f"%{{x}} : %{{y:+,.2f}} {unit.label} · %{{customdata:+.1%}} du budget du mois<extra></extra>",
    ))
    _base(fig, 340, legend=False, hovermode="closest")
    fig.update_layout(bargap=0.45)
    lo, hi = min(0.0, float(np.nanmin(unit(gaps)))), max(0.0, float(np.nanmax(unit(gaps))))
    pad = (hi - lo) * 0.18 or 1
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text=f"Écart moyen ({unit.label})", zeroline=True, zerolinewidth=1,
                     range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)])
    return fig


def gap_by_category(df: pd.DataFrame, unit: Unit, c: dict, fmt) -> go.Figure:
    """Barres horizontales de l'écart par rubrique (rouge ▲ = dépassement, vert ▼ = sous-consommation)."""
    d = df.sort_values("variance")
    x = unit(d["variance"])
    arrows = ["▲" if v > 0 else ("▼" if v < 0 else "•") for v in d["variance"]]
    fig = go.Figure(go.Bar(
        y=short_labels(d["label"], 24), x=x, orientation="h", marker_color=_signed_colors(d["variance"], c),
        customdata=list(zip(d["label"].astype(str), d["var_pct"].astype(float))),
        text=[f"{a} {fmt(abs(v))}" for a, v in zip(arrows, d["variance"])], textposition="outside",
        cliponaxis=False, constraintext="none", textfont=dict(size=11),
        hovertemplate=(f"%{{customdata[0]}}<br>Écart : %{{x:+,.3f}} {unit.label}"
                       " · %{customdata[1]:+.1%} du budget<extra></extra>"),
    ))
    _base(fig, max(240, 30 * len(d) + 50), legend=False, hovermode="closest")
    lo, hi = min(0.0, float(np.nanmin(x))), max(0.0, float(np.nanmax(x)))
    span = (hi - lo) or 1
    fig.update_layout(bargap=0.38)
    fig.update_xaxes(range=[lo - (span * 0.4 if lo < 0 else 0), hi + span * 0.4], zeroline=True,
                     zerolinewidth=1, showgrid=True, title_text=f"Écart ({unit.label})")
    fig.update_yaxes(showgrid=False)
    return fig


# --------------------------------------------------------------------------- taux de gestion

def rates_by_year(ind: pd.DataFrame, rows: list[tuple[str, str, str]], years: list, fc_years: list, c: dict,
                  ref: float = 100.0, ref_label: str = "", height: int = 340) -> go.Figure:
    """Taux (en %) par année, une série par ligne (clé, nom, couleur) ; points décalés pour rester lisibles
    même quand les valeurs se confondent ; années projetées en pointillés."""
    fig = go.Figure()
    xs, xf = [str(y) for y in years], [f"{y} P" for y in fc_years]
    values = []
    for key, name, color in rows:
        y_hist = (ind.loc[key, years] * 100).to_numpy(float) if years else np.array([])
        fig.add_scatter(x=xs, y=y_hist, name=name, mode="lines+markers", offsetgroup=key, legendgroup=key,
                        line=dict(color=color, width=2), marker=dict(size=8, color=color),
                        hovertemplate=f"{name} · %{{x}} : %{{y:.1f}} %<extra></extra>")
        values += list(y_hist)
        if fc_years:
            y_fc = (ind.loc[key, fc_years] * 100).to_numpy(float)
            fx = ([xs[-1]] if len(xs) else []) + xf
            fy = ([y_hist[-1]] if len(xs) else []) + list(y_fc)
            fig.add_scatter(x=fx, y=fy, name=f"{name} (projection)", mode="lines+markers", offsetgroup=key,
                            legendgroup=key, showlegend=False, line=dict(color=color, width=2, dash="dash"),
                            marker=dict(size=8, symbol="circle-open", color=color, line=dict(width=2)),
                            hovertemplate=f"{name} · %{{x}} : %{{y:.1f}} %<extra></extra>")
            values += list(y_fc)
    fig.add_hline(y=ref, line=dict(color=c["muted"], width=1), annotation_text=ref_label,
                  annotation_position="top left", annotation_font=dict(size=11, color=c["muted"]))
    if fc_years:
        _forecast_zone(fig, len(xs) - 0.5, len(xs) + len(xf) - 0.5, c)
    _base(fig, height)
    fig.update_layout(scattermode="group", scattergap=0.55)
    finite = [v for v in values if np.isfinite(v)] + [ref]
    lo, hi = min(finite), max(finite)
    pad = max((hi - lo) * 0.2, 1.0)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=xs + xf)
    fig.update_yaxes(ticksuffix=" %", range=[lo - pad, hi + pad])
    return fig


def rate_by_category(df: pd.DataFrame, c: dict, year: int) -> go.Figure:
    """Taux de réalisation par rubrique, en barres partant de 100 % (▲ rouge = dépassement, ▼ vert = en dessous)."""
    d = df.dropna(subset=["rate"]).sort_values("rate")
    delta = ((d["rate"] - 1) * 100).to_numpy(float)
    arrows = ["▲" if v > 0 else ("▼" if v < 0 else "•") for v in delta]
    fig = go.Figure(go.Bar(
        y=short_labels(d["label"], 26), x=delta, base=100, orientation="h", marker_color=_signed_colors(delta, c),
        text=[f"{a} {num(r * 100, 1)} %" for a, r in zip(arrows, d["rate"])], textposition="outside",
        cliponaxis=False, constraintext="none", textfont=dict(size=11),
        customdata=list(zip(d["label"].astype(str), (d["rate"] * 100).astype(float))),
        hovertemplate="%{customdata[0]} : %{customdata[1]:.1f} % du budget<extra></extra>",
    ))
    fig.add_vline(x=100, line=dict(color=c["muted"], width=1))
    _base(fig, max(240, 30 * len(d) + 50), legend=False, hovermode="closest")
    span = max(float(np.nanmax(np.abs(delta))) if len(delta) else 1.0, 1.0)
    left = 100 - span * (1.5 if (delta < 0).any() else 0.25)
    right = 100 + span * (1.5 if (delta > 0).any() else 0.25)
    fig.update_layout(bargap=0.38)
    fig.update_xaxes(range=[left, right], ticksuffix=" %", showgrid=True, title_text=f"Taux de réalisation {year}")
    fig.update_yaxes(showgrid=False)
    return fig


def quarterly_rates(hist: pd.DataFrame, fc: pd.DataFrame | None, c: dict) -> go.Figure:
    """Taux de réalisation par trimestre, une ligne par année (récente = plus marquée), projection en pointillés."""
    fyears = sorted(fc["year"].unique())[:1] if fc is not None and len(fc) else []
    years = sorted(hist["year"].unique())[-(MAX_PROFILE_YEARS - len(fyears)):]
    ramp = c["years"][max(1, len(years) + len(fyears))]
    quarters = sorted(set(hist["quarter"]) | (set(fc["quarter"]) if fyears else set()))
    x = [f"T{q}" for q in quarters]
    fig = go.Figure()
    for i, year in enumerate(years + fyears):
        is_fc = year in fyears
        src = fc if is_fc else hist
        y = src[src["year"] == year].set_index("quarter")["rate"].reindex(quarters) * 100
        name = f"{year} (projection)" if is_fc else str(year)
        fig.add_scatter(x=x, y=y, name=name, mode="lines+markers",
                        line=dict(color=ramp[i], width=2, dash="dash" if is_fc else "solid"),
                        marker=dict(size=7, color=ramp[i], symbol="circle-open" if is_fc else "circle",
                                    line=dict(width=2, color=ramp[i])),
                        hovertemplate=f"{name} · %{{x}} : %{{y:.1f}} %<extra></extra>")
    fig.add_hline(y=100, line=dict(color=c["muted"], width=1))
    _base(fig, 340)
    values = pd.concat([hist["rate"], fc["rate"] if fyears else pd.Series(dtype=float)]).dropna() * 100
    lo, hi = min(values.min(), 100), max(values.max(), 100)
    pad = max((hi - lo) * 0.2, 1.0)
    fig.update_xaxes(type="category")
    fig.update_yaxes(ticksuffix=" %", range=[lo - pad, hi + pad], title_text="Taux de réalisation")
    return fig


def consumption_curve(hist: pd.DataFrame, fc: pd.DataFrame | None, c: dict, pace_year: int | None) -> go.Figure:
    """Courbe de consommation du budget annuel : réalisé cumulé / budget de l'année, mois par mois.

    En gris : le rythme budgété (budget cumulé) de `pace_year` et le rythme linéaire (1/12 par mois).
    """
    fyears = sorted(fc["year"].unique())[:1] if fc is not None and len(fc) else []
    years = sorted(hist["year"].unique())[-(MAX_PROFILE_YEARS - len(fyears)):]
    ramp = c["years"][max(1, len(years) + len(fyears))]
    months = list(range(1, 13))
    x = [MONTH_ABBR[m - 1] for m in months]
    fig = go.Figure()
    fig.add_scatter(x=x, y=[m / 12 * 100 for m in months], name="Rythme linéaire", mode="lines",
                    line=dict(color=_rgba(c["muted"], 0.55), width=1.5),
                    hovertemplate="Rythme linéaire · %{x} : %{y:.1f} %<extra></extra>")
    if pace_year is not None and pace_year in set(hist["year"]):
        pace = hist[hist["year"] == pace_year].set_index("month")["budget_pace"].reindex(months) * 100
        fig.add_scatter(x=x, y=pace, name=f"Rythme budgété {pace_year}", mode="lines",
                        line=dict(color=c["planned"], width=2),
                        hovertemplate=f"Budget cumulé {pace_year} · %{{x}} : %{{y:.1f}} %<extra></extra>")
    for i, year in enumerate(years + fyears):
        is_fc = year in fyears
        src = fc if is_fc else hist
        y = src[src["year"] == year].set_index("month")["consumed"].reindex(months) * 100
        name = f"{year} (projection)" if is_fc else str(year)
        fig.add_scatter(x=x, y=y, name=name, mode="lines+markers",
                        line=dict(color=ramp[i], width=2, dash="dash" if is_fc else "solid"),
                        marker=dict(size=6, color=ramp[i], symbol="circle-open" if is_fc else "circle",
                                    line=dict(width=2, color=ramp[i])),
                        hovertemplate=f"{name} · fin %{{x}} : %{{y:.1f}} % du budget annuel<extra></extra>")
    fig.add_hline(y=100, line=dict(color=c["muted"], width=1), annotation_text="Budget annuel",
                  annotation_position="top left", annotation_font=dict(size=11, color=c["muted"]))
    _base(fig, 380)
    fig.update_xaxes(type="category")
    fig.update_yaxes(ticksuffix=" %", rangemode="tozero", title_text="Consommé / budget annuel")
    return fig


def growth_by_category(df: pd.DataFrame, c: dict, color: str, year: int) -> go.Figure:
    """Taux d'évolution du réalisé par rubrique (une série, une couleur), valeur au bout de la barre."""
    d = df.dropna(subset=["growth"]).sort_values("growth")
    x = (d["growth"] * 100).to_numpy(float)
    fig = go.Figure(go.Bar(
        y=short_labels(d["label"], 26), x=x, orientation="h", marker_color=color,
        text=[f"{num(v, 1, True)} %" for v in x], textposition="outside", cliponaxis=False,
        constraintext="none", textfont=dict(size=11), customdata=d["label"].astype(str),
        hovertemplate="%{customdata} : %{x:+.1f} %<extra></extra>",
    ))
    _base(fig, max(240, 30 * len(d) + 50), legend=False, hovermode="closest")
    lo, hi = min(0.0, float(np.nanmin(x)) if len(x) else 0.0), max(0.0, float(np.nanmax(x)) if len(x) else 1.0)
    span = (hi - lo) or 1.0
    fig.update_layout(bargap=0.38)
    fig.update_xaxes(range=[lo - (span * 0.35 if lo < 0 else 0), hi + span * 0.35], ticksuffix=" %", zeroline=True,
                     zerolinewidth=1, showgrid=True, title_text=f"Évolution du réalisé {year - 1} → {year}")
    fig.update_yaxes(showgrid=False)
    return fig


def structure_bars(df: pd.DataFrame, parts: list[tuple[str, str]], c: dict, fc_labels: set[str]) -> go.Figure:
    """Structure en % (barres empilées à 100 %), une barre par année ; années projetées plus claires."""
    fig = go.Figure()
    for i, (col, name) in enumerate(parts):
        color = c["cat"][i % len(c["cat"])]
        values = (df[col] * 100).to_numpy(float)
        fig.add_bar(y=df["label"], x=values, name=name, orientation="h",
                    marker_color=[_rgba(color, 0.45) if lbl in fc_labels else color for lbl in df["label"]],
                    text=[f"{num(v, 1)} %" for v in values], textposition="inside", insidetextanchor="middle",
                    textfont=dict(size=11, color="#ffffff"),
                    hovertemplate=f"{name} · %{{y}} : %{{x:.1f}} %<extra></extra>")
    _base(fig, max(220, 46 * len(df) + 70), hovermode="closest")
    fig.update_layout(barmode="stack", bargap=0.42, uniformtext=dict(minsize=10, mode="hide"))
    fig.update_xaxes(range=[0, 100], ticksuffix=" %", showgrid=True)
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    return fig
