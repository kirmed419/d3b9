"""Load, clean and repair the Budget / Réalisation workbook.

Expected layout (one sheet per scenario and year, e.g. "Budget 2024", "Realisation 2024"):

    CAPEX | <annual> | janv. | févr. | ... | déc.      <- block header
    <category> | annual amount | 12 monthly amounts     <- one row per category
    (blank)    | block total   | 12 monthly totals      <- total row (used for validation)
    OPEX  | ...                                         <- second block, same shape

Amounts may be stored as text ("117,920,000.00"). Year and scenario are read from the
sheet name because the header cells are not reliable (every sheet says "BUDGET 2023").
"""
from __future__ import annotations

import datetime as dt
import io
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PLANNED, REALISED = "Planned", "Realised"
SCENARIOS = (PLANNED, REALISED)
CAPEX, OPEX = "CAPEX", "OPEX"
BLOCKS = (CAPEX, OPEX)
BLOCK_NAMES = {CAPEX: "Investment spending (CAPEX)", OPEX: "Consumption spending (OPEX)"}

MONTHS = list(range(1, 13))
MONTH_ABBR = ["Janv", "Févr", "Mars", "Avr", "Mai", "Juin", "Juil", "Août", "Sept", "Oct", "Nov", "Déc"]
MONTH_NAMES = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
               "août", "septembre", "octobre", "novembre", "décembre"]
MCOLS = [f"m{m}" for m in MONTHS]

REPAIR_METHODS = {
    "rephase": "Re-phase like the clean years",
    "scale": "Scale months to the annual total",
    "raw": "Keep raw monthly values",
}

# Cleaned display labels (French, English), keyed by a normalised form of the original label.
_LABELS = {
    "camions": ("Camions", "Trucks"),
    "equipements": ("Équipements", "Equipment"),
    "equipementsmobiles": ("Équipements mobiles", "Mobile equipment"),
    "equipementssmobiles": ("Équipements mobiles", "Mobile equipment"),
    "infrastructure": ("Infrastructure", "Infrastructure"),
    "installationtechnique": ("Installations techniques", "Technical installations"),
    "logicielslicencesit": ("Logiciels & licences IT", "Software & IT licences"),
    "logicielslicencesiterp": ("Logiciels & licences ERP", "ERP software & licences"),
    "materielaudiovis": ("Matériel audiovisuel", "Audiovisual equipment"),
    "materielit": ("Matériel IT", "IT hardware"),
    "ouvragesdinfrastructure": ("Ouvrages d'infrastructure", "Infrastructure works"),
    "ouveragesdinfrastructure": ("Ouvrages d'infrastructure", "Infrastructure works"),
    "vhl": ("Véhicules légers (VHL)", "Light vehicles"),
    "mp": ("Matières premières (MP)", "Raw materials"),
    "fourniture": ("Fournitures", "Supplies"),
    "fournitures": ("Fournitures", "Supplies"),
    "carburantsetlubrifiants": ("Carburants & lubrifiants", "Fuel & lubricants"),
    "entrereparmaint": ("Entretien, réparation & maintenance", "Maintenance & repairs"),
    "locations": ("Locations", "Rentals"),
    "piecesrechange": ("Pièces de rechange", "Spare parts"),
    "redevancesbrevetlicenceslogiciels": ("Redevances, brevets & licences", "Royalties, patents & licences"),
    "remdinterethonoraires": ("Rémunérations d'intermédiaires & honoraires", "Intermediary & professional fees"),
    "soustraitance": ("Sous-traitance", "Subcontracting"),
    "transports": ("Transports", "Transport"),
}

_MONTH_PREFIXES = [
    (1, ("janv", "jan")), (2, ("fevr", "fev", "feb")), (3, ("mars", "mar")),
    (4, ("avr", "apr")), (5, ("mai", "may")), (6, ("juin", "jun")),
    (7, ("juil", "jul")), (8, ("aout", "aou", "aug")), (9, ("sept", "sep")),
    (10, ("oct",)), (11, ("nov",)), (12, ("dec",)),
]
_SHEET_RE = re.compile(r"^\s*(?P<kind>.*?)[\s_\-.]*(?P<year>(?:19|20)\d{2})\s*$")


@dataclass
class BudgetData:
    long: pd.DataFrame          # one row per scenario/year/block/category/month (repaired amounts)
    lines: pd.DataFrame         # one row per scenario/year/block/category, wide months, raw values
    totals: pd.DataFrame        # the workbook's own total rows, wide months
    checks: list[dict]          # data-quality findings: level ("ok" | "warn" | "info"), title, detail
    repair_log: pd.DataFrame    # line-years whose months were repaired
    multipliers: pd.DataFrame   # each sheet's total as a multiple of the first plan
    sheets: pd.DataFrame        # sheet name -> scenario, year, header labels
    source_name: str
    repair_method: str
    notes: dict = field(default_factory=dict)

    @property
    def years(self) -> list[int]:
        return sorted(self.long["year"].unique().tolist())

    def categories(self, block: str) -> pd.DataFrame:
        cats = self.long.loc[self.long["block"] == block, ["category", "label_fr", "label_en", "order"]]
        return cats.drop_duplicates("category").sort_values("order").reset_index(drop=True)


# --------------------------------------------------------------------------- parsing helpers

def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def clean_label(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()
    return text or None


def category_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _strip_accents(label).lower())


def display_labels(category: str) -> tuple[str, str]:
    return _LABELS.get(category_key(category), (category, category))


def to_number(value) -> float:
    """Parse a cell that may hold a number or a formatted string ("1,234.56", "1 234,56", "(12)")."""
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    for ch in ("\xa0", " ", " ", "'"):
        text = text.replace(ch, "")
    if text in ("", "-", "–", "—"):
        return np.nan
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        text = text.replace(",", "") if len(parts) > 2 or len(parts[-1]) == 3 else text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return np.nan
    return -number if negative else number


def parse_sheet_name(name: str) -> tuple[str, int] | None:
    match = _SHEET_RE.match(str(name))
    if not match:
        return None
    kind = category_key(match["kind"])
    year = int(match["year"])
    if re.match(r"^(budget|budg|prevu|prev|plan|objectif|target)", kind):
        return PLANNED, year
    if re.match(r"^(realis|realiz|real|actual|reel)", kind):
        return REALISED, year
    return None


def _parse_month(value) -> tuple[int | None, int | None]:
    """Return (month, year) for a header cell such as "janv.-23", "Feb 2024" or a date."""
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.month, value.year
    text = clean_label(value)
    if not text:
        return None, None
    key = _strip_accents(text).lower()
    word = re.match(r"[a-z]+", key)
    if not word or len(word.group(0)) > 9:
        return None, None
    for month, prefixes in _MONTH_PREFIXES:
        if word.group(0).startswith(prefixes):
            year = re.search(r"(\d{2,4})\s*$", key)
            yr = int(year.group(1)) if year else None
            if yr is not None and yr < 100:
                yr += 2000
            return month, yr
    return None, None


def _detect_columns(header: list) -> dict:
    months: dict[int, int] = {}
    header_years: set[int] = set()
    annual_col, annual_header = None, None
    for j, cell in enumerate(header[1:], start=1):
        month, year = _parse_month(cell)
        if month and month not in months:
            months[month] = j
            if year:
                header_years.add(year)
        elif annual_col is None and clean_label(cell):
            annual_col, annual_header = j, clean_label(cell)
    if len(months) < 12:  # fall back to the standard layout: B = annual, C..N = Jan..Dec
        months = {m: m + 1 for m in MONTHS}
        annual_col = 1
    if annual_header:
        found = re.search(r"((?:19|20)\d{2})", annual_header)
        if found:
            header_years.add(int(found.group(1)))
    return {"months": months, "annual": annual_col, "annual_header": annual_header or "", "years": header_years}


def _cell(row: list, j: int | None):
    return row[j] if j is not None and j < len(row) else None


# --------------------------------------------------------------------------- loading

def _read_sheets(file_bytes: bytes):
    book = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None, dtype=object, engine="openpyxl")
    lines, totals, sheets, skipped = [], [], [], []
    text_numbers = 0
    for name, raw in book.items():
        parsed = parse_sheet_name(name)
        if parsed is None:
            skipped.append(name)
            continue
        scenario, year = parsed
        block, cols = None, None
        seen: set[tuple[str, str]] = set()
        header_info = {"sheet": name, "scenario": scenario, "year": year, "header_labels": [], "header_years": set()}
        for i in range(len(raw)):
            row = raw.iloc[i].tolist()
            label = clean_label(row[0]) if row else None
            if label and label.upper() in BLOCKS:
                block, cols = label.upper(), _detect_columns(row)
                header_info["header_labels"].append(cols["annual_header"])
                header_info["header_years"] |= cols["years"]
                continue
            if block is None:
                continue
            cells = [_cell(row, cols["annual"])] + [_cell(row, cols["months"][m]) for m in MONTHS]
            values = [to_number(c) for c in cells]
            text_numbers += sum(isinstance(c, str) and not np.isnan(v) for c, v in zip(cells, values))
            if all(np.isnan(v) for v in values):
                continue
            record = {"scenario": scenario, "year": year, "block": block, "annual": values[0],
                      **dict(zip(MCOLS, values[1:]))}
            if label is None:
                totals.append(record)
                continue
            category = label
            while (block, category) in seen:  # keep duplicate rows distinct
                category += " (bis)"
            seen.add((block, category))
            lines.append({**record, "category": category, "order": i})
        sheets.append(header_info)
    return pd.DataFrame(lines), pd.DataFrame(totals), sheets, skipped, text_numbers


def _flag_mismatches(lines: pd.DataFrame) -> pd.DataFrame:
    lines = lines.copy()
    lines["month_sum"] = lines[MCOLS].fillna(0).sum(axis=1)
    no_annual = lines["annual"].isna()
    lines.loc[no_annual, "annual"] = lines.loc[no_annual, "month_sum"]
    tolerance = np.maximum(1.0, 1e-6 * lines["annual"].abs())
    lines["status"] = np.where((lines["month_sum"] - lines["annual"]).abs() <= tolerance, "ok", "mismatch")
    lines.loc[no_annual, "status"] = "no annual"
    return lines


def _repair(lines: pd.DataFrame, method: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Return repaired monthly values for every line plus a log of what was changed.

    "rephase" spreads the annual amount with the monthly pattern of the closest clean year of the
    same category (same scenario first). "scale" keeps the raw monthly shape and rescales it.
    """
    raw = lines[MCOLS].fillna(0).to_numpy(float)
    fixed = raw.copy()
    log = []
    clean = lines[(lines["status"] != "mismatch") & (lines["month_sum"] > 0)]
    for idx in np.flatnonzero(lines["status"].eq("mismatch").to_numpy()):
        row = lines.iloc[idx]
        profile, reference = None, None
        if method == "raw":
            reference = "kept as in file"
        else:
            if method == "rephase":
                cand = clean[(clean["block"] == row["block"]) & (clean["category"] == row["category"])]
                if len(cand):
                    cand = cand.assign(_dist=(cand["year"] - row["year"]).abs(),
                                       _other=(cand["scenario"] != row["scenario"]).astype(int))
                    best = cand.sort_values(["_dist", "_other", "year"]).iloc[0]
                    profile = best[MCOLS].fillna(0).to_numpy(float) / best["month_sum"]
                    reference = f"phased like {best['scenario']} {best['year']}"
            if profile is None:
                if row["month_sum"] > 0:
                    profile, reference = raw[idx] / row["month_sum"], "months scaled to annual"
                else:
                    profile, reference = np.full(12, 1 / 12), "spread evenly"
            fixed[idx] = row["annual"] * profile
        log.append({
            "scenario": row["scenario"], "year": row["year"], "block": row["block"], "category": row["category"],
            "annual": row["annual"], "raw_month_sum": row["month_sum"],
            "months_vs_annual": row["month_sum"] / row["annual"] if row["annual"] else np.nan,
            "fix": reference,
        })
    return fixed, pd.DataFrame(log)


def _multipliers(lines: pd.DataFrame) -> pd.DataFrame:
    """Express each sheet as a multiple of the first plan; detect uniform scaling across lines."""
    plans = lines[lines["scenario"] == PLANNED]
    if plans.empty:
        return pd.DataFrame()
    ref_year = plans["year"].min()
    ref = plans[plans["year"] == ref_year].set_index(["block", "category"])["annual"]
    out = []
    for (scenario, year), g in lines.groupby(["scenario", "year"], sort=False):
        ratios = g.set_index(["block", "category"])["annual"] / ref.reindex(g.set_index(["block", "category"]).index)
        ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
        total_ratio = g["annual"].sum() / ref.sum() if ref.sum() else np.nan
        spread = (ratios.max() - ratios.min()) if len(ratios) else np.nan
        out.append({"sheet": f"{'Budget' if scenario == PLANNED else 'Realisation'} {year}", "scenario": scenario,
                    "year": year, "total": g["annual"].sum(), "multiplier": total_ratio,
                    "line_spread": spread, "uniform": bool(len(ratios)) and spread < 1e-6})
    df = pd.DataFrame(out).sort_values(["year", "scenario"]).reset_index(drop=True)
    df["reference"] = f"Budget {ref_year}"
    return df


def _quality_checks(lines, totals, fixed_long, sheets, skipped, text_numbers, log, multipliers, method):
    checks = []
    names = ", ".join(s["sheet"] for s in sheets)
    checks.append({"level": "ok", "title": f"{len(sheets)} sheets read", "detail": names})
    if skipped:
        checks.append({"level": "warn", "title": f"{len(skipped)} sheet(s) ignored",
                       "detail": "Name does not look like 'Budget YYYY' / 'Realisation YYYY': " + ", ".join(skipped)})

    stale = [s for s in sheets if s["header_years"] and s["header_years"] != {s["year"]}]
    wrong_kind = [s for s in sheets if s["scenario"] == REALISED
                  and any("budget" in h.lower() for h in s["header_labels"])]
    if stale or wrong_kind:
        sample = next(iter(sorted(stale[0]["header_years"]))) if stale else None
        detail = (f"{len(stale)} sheet(s) carry header dates for another year"
                  + (f" (e.g. {sample})" if sample else "")
                  + (f", and {len(wrong_kind)} 'Realisation' sheet(s) are headed 'BUDGET'" if wrong_kind else "")
                  + ". Year and scenario are taken from the sheet names instead.")
        checks.append({"level": "info", "title": "Header labels are copy-pasted", "detail": detail})

    if text_numbers:
        checks.append({"level": "info", "title": f"{text_numbers:,} amounts stored as text",
                       "detail": "Values such as '117,920,000.00' were converted to numbers."})

    if len(log):
        order = lines.drop_duplicates("category").set_index("category")["order"]
        items = sorted(log["category"].unique(), key=lambda c: order.get(c, 0))
        years = sorted(log["year"].unique())
        lo, hi = log["months_vs_annual"].min(), log["months_vs_annual"].max()
        span = f"{lo:.0f}×" if round(lo) == round(hi) else f"{lo:.0f}–{hi:.0f}×"
        how = {"rephase": "re-phased with the month pattern of the closest clean year",
               "scale": "rescaled so the months add up to the annual column",
               "raw": "left as-is (monthly totals will NOT match the annual column)"}[method]
        checks.append({
            "level": "warn",
            "title": f"{len(log)} line-years: months don't add up to the annual total",
            "detail": (f"{', '.join(display_labels(c)[0] for c in items)} in {', '.join(map(str, years))}: "
                       f"the twelve months sum to {span} the annual column - the one-off amount was copied "
                       f"into every month. They were {how}."),
        })
    else:
        checks.append({"level": "ok", "title": "Monthly values add up to the annual column", "detail": ""})

    if not totals.empty:
        month_lines = (fixed_long.groupby(["scenario", "year", "block", "month"])["amount"].sum()
                       .unstack("month").reindex(columns=MONTHS))
        file_tot = totals.groupby(["scenario", "year", "block"])[MCOLS].sum()
        file_tot.columns = MONTHS
        common = month_lines.index.intersection(file_tot.index)
        diff = (month_lines.loc[common] - file_tot.loc[common]).abs()
        tolerance = np.maximum(1.0, 1e-6 * file_tot.loc[common].abs())
        bad = int((diff > tolerance).any(axis=1).sum())
        if bad:
            checks.append({"level": "warn", "title": "Monthly totals differ from the file's total rows",
                           "detail": f"{bad} of {len(common)} block-years differ (largest gap "
                                     f"{diff.to_numpy().max():,.0f}). Switch the repair option to compare."})
        else:
            checks.append({"level": "ok", "title": "Monthly totals match the file's total rows",
                           "detail": f"All {len(common)} block-years agree to within {diff.to_numpy().max():.2f}."})

    if not multipliers.empty and multipliers["uniform"].all():
        mult = ", ".join(f"{r.sheet} ×{r.multiplier:.3f}" for r in multipliers.itertuples())
        checks.append({"level": "info", "title": "Every sheet is a uniform multiple of the first plan",
                       "detail": f"Relative to {multipliers['reference'].iat[0]}: {mult}. Every category moves "
                                 "in lockstep, so trends are perfectly regular and forecasts are very smooth."})
    return checks


def load_workbook(file_bytes: bytes, source_name: str, repair_method: str = "rephase") -> BudgetData:
    lines, totals, sheets, skipped, text_numbers = _read_sheets(file_bytes)
    if lines.empty:
        raise ValueError("No 'Budget YYYY' / 'Realisation YYYY' sheet with a CAPEX or OPEX block was found.")
    lines = _flag_mismatches(lines)
    lines["order"] = lines.groupby(["block", "category"])["order"].transform("min")
    fixed, log = _repair(lines, repair_method)

    ids = ["scenario", "year", "block", "category", "order"]
    long = lines[ids].loc[lines.index.repeat(12)].reset_index(drop=True)
    long["month"] = np.tile(MONTHS, len(lines))
    long["amount"] = fixed.reshape(-1)
    long["amount_raw"] = lines[MCOLS].to_numpy(float).reshape(-1)
    long["repaired"] = np.repeat(lines["status"].eq("mismatch").to_numpy() & (repair_method != "raw"), 12)
    long["date"] = pd.to_datetime(pd.DataFrame({"year": long["year"], "month": long["month"], "day": 1}))
    labels = {c: display_labels(c) for c in long["category"].unique()}
    long["label_fr"] = long["category"].map(lambda c: labels[c][0])
    long["label_en"] = long["category"].map(lambda c: labels[c][1])

    multipliers = _multipliers(lines)
    checks = _quality_checks(lines, totals, long, sheets, skipped, text_numbers, log, multipliers, repair_method)
    sheet_df = pd.DataFrame([{
        "sheet": s["sheet"], "scenario": s["scenario"], "year": s["year"],
        "header says": " / ".join(dict.fromkeys(s["header_labels"])),
        "header year": ", ".join(map(str, sorted(s["header_years"]))) or "-",
    } for s in sheets])
    return BudgetData(long=long, lines=lines, totals=totals, checks=checks, repair_log=log,
                      multipliers=multipliers, sheets=sheet_df, source_name=source_name,
                      repair_method=repair_method, notes={"text_numbers": text_numbers, "skipped": skipped})
