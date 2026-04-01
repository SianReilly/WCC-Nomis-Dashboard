"""
Westminster City Council - Census 2021 & IMD Analytical Dashboard
=================================================================
Author: Westminster City Council - City Intelligence & Data Team
Built with: Python, Streamlit, Plotly, Scikit-learn, Statsmodels, SciPy

Data sources:
  - IMD 2025 & 2019: MHCLG (Ministry of Housing, Communities & Local Government)
  - Census 2021 ward-level data: ONS / Nomis API (live, refreshed hourly)
  - Ward boundaries: WCC 2022 ward boundaries (18 wards, converted from GIS TopoJSON)

To run locally:   streamlit run nomis_dashboard.py
To deploy:        Push to GitHub > Streamlit Community Cloud

requirements.txt:
  streamlit, pandas, numpy, plotly, scikit-learn, statsmodels, scipy, requests

Live Nomis datasets used:
  TS066 (NM_2083_1) - Economic activity (employment & unemployment)
  TS007 (NM_2027_1) - Age by single year (total population)
  TS037 (NM_2055_1) - General health
  TS054 (NM_2072_1) - Tenure (housing)
  TS067 (NM_2084_1) - Highest level of qualification

Geography: Westminster wards, TYPE298, PARENT:1946157124
Modelled estimates are used as fallback if Nomis is unavailable,
and for variables not covered by the 5 API tables above.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json, os, io, requests, re

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error, roc_auc_score,
                              roc_curve, silhouette_score, davies_bouldin_score)
from sklearn.model_selection import (cross_val_score, LeaveOneOut,
                                     KFold, StratifiedKFold)
from scipy import stats
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="WCC Census & IMD Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# COLOUR PALETTE
# Westminster brand + Economist-influenced chart palette
# =============================================================================
WCC_BLUE   = "#003087"   # Westminster navy
WCC_GOLD   = "#C8A84B"   # Westminster gold
TEAL       = "#028090"   # Improvement/positive signal
CORAL      = "#E87461"   # Mid-range
SAGE       = "#84B59F"   # Fourth category
ECON_RED   = "#E3120B"   # High-deprivation highlight
ECON_GREY  = "#CCCCCC"   # Axis/grid grey
LIGHT_GRID = "#E8E8E8"   # Chart gridlines
DARK_TEXT  = "#1A1A1A"   # Near-black headings
MID_TEXT   = "#555555"   # Subtitles / axis labels

# Total England wards (2022 ONS boundaries) - used for national percentile bands
TOTAL_WARDS_ENGLAND = 6904

# Nomis API constants - Westminster LA geography code under Nomis
NOMIS_BASE           = "https://www.nomisweb.co.uk/api/v01/dataset"
WESTMINSTER_PARENT   = "1946157124"   # Westminster City Council LA code in Nomis


# =============================================================================
# CSS STYLING
# Economist-style typography with Libre Baskerville headlines
# and Source Sans 3 body copy. WCC brand colours throughout.
# =============================================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Source+Sans+3:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }

  .econ-header { border-top: 4px solid #003087; padding: 1.2rem 0 0.8rem; margin-bottom: 1rem; }
  .econ-header h1 { font-family: 'Libre Baskerville', serif; font-size: 1.85rem; color: #1A1A1A; margin: 0 0 0.3rem; }
  .econ-header p { color: #555; font-size: 0.9rem; margin: 0; }

  /* KPI cards - white-space:nowrap prevents long numbers like 216,972 wrapping to two lines */
  .kpi-card { background: white; border-top: 3px solid #003087; padding: 0.75rem 0.9rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .kpi-card h2 { margin: 0; font-size: 1.55rem; color: #003087; font-family: 'Libre Baskerville', serif; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .kpi-card p { margin: 0.15rem 0 0; font-size: 0.76rem; color: #555; }
  .kpi-card.red  { border-top-color: #E3120B; } .kpi-card.red  h2 { color: #E3120B; }
  .kpi-card.gold { border-top-color: #C8A84B; } .kpi-card.gold h2 { color: #C8A84B; }
  .kpi-card.teal { border-top-color: #028090; } .kpi-card.teal h2 { color: #028090; }

  .section-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #E3120B; margin: 1.1rem 0 0.1rem; }
  .section-title { font-family: 'Libre Baskerville', serif; font-size: 1.05rem; font-weight: 700; color: #1A1A1A; border-bottom: 1px solid #E8E8E8; padding-bottom: 0.3rem; margin-bottom: 0.5rem; }

  .insight-box { background: #F7F9FC; border-left: 3px solid #028090; padding: 0.6rem 0.85rem; font-size: 0.84rem; margin: 0.4rem 0; }
  .warn-box    { background: #FFFBF0; border-left: 3px solid #C8A84B; padding: 0.6rem 0.85rem; font-size: 0.84rem; margin: 0.4rem 0; }
  .alert-box   { background: #FFF5F5; border-left: 3px solid #E3120B; padding: 0.6rem 0.85rem; font-size: 0.84rem; margin: 0.4rem 0; }
  .method-box  { background: #F0F4FA; border: 1px solid #C8D8F0; padding: 0.85rem 1rem; border-radius: 4px; margin: 0.5rem 0; font-size: 0.83rem; }
  .real-box    { background: #F0FAF0; border-left: 3px solid #028090; padding: 0.5rem 0.85rem; font-size: 0.81rem; margin: 0.3rem 0; }

  .stat-pill { font-size: 0.78rem; font-family: 'Courier New', monospace; background: #EEEEEE; padding: 0.2rem 0.5rem; border-radius: 3px; display: inline-block; margin: 0.1rem; }
  .context-banner { background: #1A1A2E; color: #c8e6ff; padding: 0.55rem 0.95rem; font-size: 0.81rem; margin-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CHART THEME HELPERS
#
# econ()   - vertical charts (bar, scatter, line, heatmap)
# econ_h() - horizontal bar charts
#
# Economist house style: bold serif title, light grey gridlines,
# source note below x-axis labels. No red top rule (removed per WCC preference).
#
# rotated=True: use when x-axis labels are at -45 degrees. Increases the
# bottom margin so the source note never overlaps the angled label text.
# =============================================================================
def econ(fig, title="", subtitle="",
         src="MHCLG IMD 2025; ONS Census 2021 via Nomis",
         h=420, xgrid=False, rotated=False):
    """Economist style for vertical charts. No red top rule."""
    ttl = ""
    if title:
        ttl += f"<b style='font-family:Georgia,serif;font-size:14px'>{title}</b>"
    if subtitle:
        ttl += f"<br><span style='font-size:10px;color:{MID_TEXT}'>{subtitle}</span>"

    # Extra bottom margin when labels are rotated - source note must clear them
    b_margin = 20
    if src:
        b_margin = 105 if rotated else 72

    fig.update_layout(
        height=h,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=11, color=DARK_TEXT),
        title=dict(text=ttl, x=0, xanchor="left", y=0.98, yanchor="top",
                   pad=dict(l=0, t=8)),
        xaxis=dict(showgrid=xgrid, gridcolor=LIGHT_GRID, zeroline=False,
                   showline=True, linecolor=ECON_GREY, linewidth=1,
                   tickfont=dict(size=10, color=MID_TEXT),
                   title_font=dict(size=10, color=MID_TEXT)),
        yaxis=dict(showgrid=True, gridcolor=LIGHT_GRID, zeroline=False,
                   showline=False,
                   tickfont=dict(size=10, color=MID_TEXT),
                   title_font=dict(size=10, color=MID_TEXT)),
        margin=dict(l=10, r=10, t=75 if ttl else 28, b=b_margin),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hoverlabel=dict(bgcolor="white", font_size=11, bordercolor=ECON_GREY),
    )

    # Source note - positioned below x-axis, never overlapping labels
    # y=-0.28 when rotated, y=-0.19 otherwise
    if src:
        fig.add_annotation(
            text=f"<span style='font-size:9px;color:#888888'>Source: {src}</span>",
            xref="paper", yref="paper",
            x=0, y=(-0.28 if rotated else -0.19),
            showarrow=False, align="left", xanchor="left"
        )
    return fig


def econ_h(fig, title="", subtitle="",
           src="MHCLG IMD 2025; ONS Census 2021 via Nomis", h=420):
    """Economist style for horizontal bar charts. Gridlines on value axis."""
    fig = econ(fig, title=title, subtitle=subtitle, src=src, h=h)
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor=LIGHT_GRID, zeroline=False,
                   showline=False,
                   tickfont=dict(size=10, color=MID_TEXT),
                   title_font=dict(size=10, color=MID_TEXT)),
        yaxis=dict(showgrid=False, zeroline=False,
                   showline=True, linecolor=ECON_GREY, linewidth=1,
                   tickfont=dict(size=10, color=MID_TEXT),
                   title_font=dict(size=10, color=MID_TEXT)),
    )
    return fig


# =============================================================================
# NOMIS API HELPERS
# =============================================================================

@st.cache_data(ttl=3600)   # cache for 1 hour so live data stays fresh
def fetch_nomis(dataset_id: str, category_param: str, measures: str = "20301"):
    """
    Generic Nomis API fetcher for all Westminster 2022 wards (TYPE298).

    Parameters
    ----------
    dataset_id     : Nomis dataset ID e.g. "NM_2083_1"
    category_param : URL fragment e.g. "c2021_eastat_20=1001,1006,1011"
                     Pass empty string "" to fetch all categories.
    measures       : "20301" = percentage (default); "20100" = count

    Returns (DataFrame | None, error_string | None).
    On success, columns are uppercased and ward names normalised to match
    the canonical WCC ward list.
    """
    url = (
        f"{NOMIS_BASE}/{dataset_id}.data.csv"
        f"?date=latest"
        f"&geography=TYPE298&geography_filter=PARENT:{WESTMINSTER_PARENT}"
        f"&measures={measures}"
    )
    if category_param:
        url += f"&{category_param}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        if not resp.text.strip():
            return None, "Nomis returned empty response"

        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty:
            return None, "Nomis CSV has no data rows"

        # Normalise column headers to uppercase for consistent access
        df.columns = [c.strip().upper() for c in df.columns]

        # Normalise ward names to match WCC canonical list:
        # Strip trailing " Ward" (Nomis sometimes adds this)
        # Convert " and " -> " & " for Knightsbridge & Belgravia etc.
        if "GEOGRAPHY_NAME" in df.columns:
            df["GEOGRAPHY_NAME"] = (
                df["GEOGRAPHY_NAME"]
                .str.strip()
                .str.replace(r"\s+[Ww]ard$", "", regex=True)
                .str.replace(" and ", " & ", regex=False)
                .str.replace(" And ", " & ", regex=False)
            )
        return df, None

    except requests.exceptions.Timeout:
        return None, "Nomis API timed out (30s)"
    except requests.exceptions.HTTPError as e:
        return None, f"Nomis HTTP error: {e}"
    except pd.errors.EmptyDataError:
        return None, "Nomis returned empty CSV"
    except Exception as e:
        return None, str(e)


def find_cat_col(df: pd.DataFrame) -> str | None:
    """
    Find the category name column in a Nomis response.
    Nomis names these like C2021_EASTAT_20_NAME, C2021_HEALTH_6_NAME, etc.
    We look for any column starting with C2021_ and ending with _NAME.
    """
    for col in df.columns:
        if col.startswith("C2021_") and col.endswith("_NAME"):
            return col
    return None


# =============================================================================
# LIVE NOMIS DATA LOADER
# Pulls the 5 Census 2021 tables requested and returns a ward-level DataFrame.
# Falls back to modelled estimates if any individual API fails.
# =============================================================================

@st.cache_data(ttl=3600)
def load_nomis_census():
    """
    Load real Census 2021 ward-level data from Nomis for Westminster.

    Datasets:
      TS066 (NM_2083_1) - Employment rate and unemployment %
      TS007 (NM_2027_1) - Total population per ward
      TS037 (NM_2055_1) - Good health % (Very good + Good)
      TS054 (NM_2072_1) - Tenure (owner occupied, social rented, private rented)
      TS067 (NM_2084_1) - Qualifications (no quals %, degree level %)

    Returns
    -------
    (DataFrame with Ward column + real variables, dict of API success flags)
    """
    ward_data = {}   # dict of ward_name -> {variable: value}
    status    = {}   # which datasets successfully loaded

    def add(ward: str, key: str, value):
        """Helper to safely add a value to the ward_data dict."""
        ward_data.setdefault(ward, {})[key] = value

    # ------------------------------------------------------------------
    # TS066: Economic activity (NM_2083_1)
    # Categories: 1001=In employment, 1006=Unemployed, 1011=Economically inactive
    # We request percentages (20301) - Nomis pre-calculates % of 16+ residents
    # ------------------------------------------------------------------
    df66, err66 = fetch_nomis("NM_2083_1", "c2021_eastat_20=1001,1006,1011", "20301")
    status["TS066"] = df66 is not None and "GEOGRAPHY_NAME" in (df66.columns if df66 is not None else [])

    if status["TS066"]:
        cat66 = find_cat_col(df66)
        if cat66:
            for ward, grp in df66.groupby("GEOGRAPHY_NAME"):
                # Employment = any row whose category name contains "In employment"
                emp_rows = grp[grp[cat66].str.contains("In employment", case=False, na=False)]
                une_rows = grp[grp[cat66].str.contains("Unemployed", case=False, na=False)]
                if not emp_rows.empty:
                    add(ward, "Employment Rate", round(float(emp_rows["OBS_VALUE"].sum()), 1))
                if not une_rows.empty:
                    add(ward, "Unemployment %",  round(float(une_rows["OBS_VALUE"].sum()), 1))
        else:
            status["TS066"] = False   # no category name column - unusable response

    # ------------------------------------------------------------------
    # TS007: Age by single year (NM_2027_1)
    # We request all categories (no filter) with counts (20100)
    # then find the "All usual residents" total row per ward.
    # If no total row found, sum all individual age counts.
    # ------------------------------------------------------------------
    df07, err07 = fetch_nomis("NM_2027_1", "", "20100")
    status["TS007"] = df07 is not None and "GEOGRAPHY_NAME" in (df07.columns if df07 is not None else [])

    if status["TS007"]:
        cat07 = find_cat_col(df07)
        for ward, grp in df07.groupby("GEOGRAPHY_NAME"):
            # Look for a "Total" or "All" row first
            if cat07:
                total_rows = grp[grp[cat07].str.lower().str.contains(
                    r"all usual|all ages|total", na=False)]
            else:
                total_rows = pd.DataFrame()

            if not total_rows.empty:
                pop = int(total_rows["OBS_VALUE"].max())
            else:
                # No total row - sum everything (individual single years don't overlap)
                pop = int(grp["OBS_VALUE"].sum())

            add(ward, "Population", pop)

    # ------------------------------------------------------------------
    # TS037: General health (NM_2055_1)
    # Categories 0=All, 1=Very good, 2=Good, 3=Fair, 4=Bad, 5=Very bad
    # Good Health % = sum of "Very good health" + "Good health" percentages
    # As a bonus, derive a Long-term Illness proxy from "Bad" + "Very bad"
    # ------------------------------------------------------------------
    df37, err37 = fetch_nomis("NM_2055_1", "c2021_health_6=0...5", "20301")
    status["TS037"] = df37 is not None and "GEOGRAPHY_NAME" in (df37.columns if df37 is not None else [])

    if status["TS037"]:
        cat37 = find_cat_col(df37)
        if cat37:
            for ward, grp in df37.groupby("GEOGRAPHY_NAME"):
                # Good health = "Very good health" + "Good health" rows
                good = grp[grp[cat37].str.contains(
                    r"Very good health|Good health", case=False, na=False)]
                # Ill health proxy = "Bad health" + "Very bad health"
                ill  = grp[grp[cat37].str.contains(
                    r"Bad health|Very bad", case=False, na=False)]
                if not good.empty:
                    add(ward, "Good Health %",       round(float(good["OBS_VALUE"].sum()), 1))
                if not ill.empty:
                    add(ward, "Long-term Illness %", round(float(ill["OBS_VALUE"].sum()), 1))
        else:
            status["TS037"] = False

    # ------------------------------------------------------------------
    # TS054: Tenure (NM_2072_1)
    # Categories include Owned outright, Owned with mortgage,
    # Social rented (council + other), Private rented
    # ------------------------------------------------------------------
    df54, err54 = fetch_nomis(
        "NM_2072_1",
        "c2021_tenure_9=0,1001...1004,8,9996,9997",
        "20301"
    )
    status["TS054"] = df54 is not None and "GEOGRAPHY_NAME" in (df54.columns if df54 is not None else [])

    if status["TS054"]:
        cat54 = find_cat_col(df54)
        if cat54:
            for ward, grp in df54.groupby("GEOGRAPHY_NAME"):
                # Owned = "Owned outright" + "Owned with a mortgage or loan"
                owner   = grp[grp[cat54].str.contains(r"Owned outright|Owned with", case=False, na=False)]
                # Social = "Social rented: Rented from council" + "Social rented: Other"
                social  = grp[grp[cat54].str.contains(r"Social rented", case=False, na=False)]
                # Private rented (excludes social rented)
                private = grp[grp[cat54].str.contains(r"Private rented", case=False, na=False)]
                if not owner.empty:
                    add(ward, "Owner Occupied %",  round(float(owner["OBS_VALUE"].sum()),  1))
                if not social.empty:
                    add(ward, "Social Rented %",   round(float(social["OBS_VALUE"].sum()),  1))
                if not private.empty:
                    add(ward, "Private Rented %",  round(float(private["OBS_VALUE"].sum()), 1))
        else:
            status["TS054"] = False

    # ------------------------------------------------------------------
    # TS067: Highest level of qualification (NM_2084_1)
    # Only counts available (measures=20100) - calculate % manually
    # No Qualifications % = "No qualifications" count / ward total * 100
    # Degree Level %      = "Level 4 qualifications and above" / total * 100
    # ------------------------------------------------------------------
    df67, err67 = fetch_nomis("NM_2084_1", "c2021_hiqual_8=1...7", "20100")
    status["TS067"] = df67 is not None and "GEOGRAPHY_NAME" in (df67.columns if df67 is not None else [])

    if status["TS067"]:
        cat67 = find_cat_col(df67)
        if cat67:
            for ward, grp in df67.groupby("GEOGRAPHY_NAME"):
                total = grp["OBS_VALUE"].sum()
                if total > 0:
                    nq = grp[grp[cat67].str.contains(r"No qualifications", case=False, na=False)]["OBS_VALUE"].sum()
                    dg = grp[grp[cat67].str.contains(r"Level 4",           case=False, na=False)]["OBS_VALUE"].sum()
                    if nq > 0:
                        add(ward, "No Qualifications %", round(float(nq / total * 100), 1))
                    if dg > 0:
                        add(ward, "Degree Level %",      round(float(dg / total * 100), 1))
        else:
            status["TS067"] = False

    # Build output DataFrame
    if ward_data:
        out = pd.DataFrame.from_dict(ward_data, orient="index").reset_index()
        out.rename(columns={"index": "Ward"}, inplace=True)
        # Only trust Nomis results when we got data for most wards (>=15 of 18)
        for api, ok in status.items():
            if ok:
                n_wards_with_data = len(ward_data)
                if n_wards_with_data < 15:
                    status[api] = False   # too few wards - something went wrong
        return out, status

    return pd.DataFrame(columns=["Ward"]), {k: False for k in status}


# =============================================================================
# MODELLED CENSUS ESTIMATES (fallback / supplementary)
# These are synthetic variables anchored to real IMD ward ranks.
# Used when Nomis API is unavailable and for variables not in the 5 tables.
# =============================================================================

def _build_modelled_estimates(df_imd: pd.DataFrame) -> pd.DataFrame:
    """
    Generate modelled census estimates for all WCC wards.

    All variables are statistically anchored to real IMD 2025 scores using
    a monotone relationship plus controlled noise (seed=42 for reproducibility).
    Higher IMD score = more deprived = direction of all modelled relationships.

    NOT real Census 2021 data. Used as fallback only.
    """
    n  = len(df_imd)
    mn = df_imd["IMD 2025 Score"].min()
    mx = df_imd["IMD 2025 Score"].max()
    # d = 0 (least deprived) to 1 (most deprived)
    d  = (df_imd["IMD 2025 Score"].values - mn) / (mx - mn)
    np.random.seed(42)   # fixed seed - same estimates every run

    pop        = np.random.randint(7_500, 15_500, n)
    emp        = (80 - d * 22 + np.random.normal(0, 1.5, n)).clip(52, 82).round(1)
    unemp      = ( 3 + d * 10 + np.random.normal(0, 0.8, n)).clip(1.5, 15).round(1)
    no_qual    = ( 4 + d * 18 + np.random.normal(0, 1.2, n)).clip(2, 26).round(1)
    degree     = (70 - d * 35 + np.random.normal(0, 2.5, n)).clip(28, 75).round(1)
    social     = ( 5 + d * 52 + np.random.normal(0, 2.5, n)).clip(3, 62).round(1)
    owner      = (58 - d * 38 + np.random.normal(0, 2.0, n)).clip(8, 62).round(1)
    private    = (100 - social - owner).clip(5, 75).round(1)
    white      = (82 - d * 38 + np.random.normal(0, 3.0, n)).clip(30, 88).round(1)
    asian      = ( 5 + d * 18 + np.random.normal(0, 1.5, n)).clip(2, 30).round(1)
    black      = ( 2 + d * 16 + np.random.normal(0, 1.5, n)).clip(1, 22).round(1)
    mixed      = (100 - white - asian - black).clip(2, 18).round(1)
    good_hlth  = (88 - d * 22 + np.random.normal(0, 1.5, n)).clip(60, 92).round(1)
    long_ill   = ( 5 + d * 12 + np.random.normal(0, 1.0, n)).clip(3, 20).round(1)
    avg_age    = (44 -  d *  8 + np.random.normal(0, 1.2, n)).clip(30, 48).round(1)
    overcrowd  = ( 2 + d * 20 + np.random.normal(0, 1.5, n)).clip(1, 26).round(1)
    med_rooms  = (4.5 - d *  2 + np.random.normal(0, 0.2, n)).clip(2.0, 5.5).round(1)

    return pd.DataFrame({
        "Ward":                 df_imd["Ward"].values,
        "Population":          pop,
        "Employment Rate":     emp,
        "Unemployment %":      unemp,
        "No Qualifications %": no_qual,
        "Degree Level %":      degree,
        "Social Rented %":     social,
        "Owner Occupied %":    owner,
        "Private Rented %":    private,
        "White %":             white,
        "Asian %":             asian,
        "Black %":             black,
        "Mixed %":             mixed,
        "Good Health %":       good_hlth,
        "Long-term Illness %": long_ill,
        "Average Age":         avg_age,
        "Overcrowding %":      overcrowd,
        "Median Rooms":        med_rooms,
    })


# =============================================================================
# MAIN DATA LOADER
# Combines real IMD data with live Nomis census data (or modelled fallback).
# Returns (DataFrame, list_of_real_columns).
# =============================================================================

@st.cache_data(ttl=3600)
def load_imd_and_census():
    """
    Build the main analytical dataset for the dashboard.

    1. Creates the IMD base table with real 2025 and 2019 scores/ranks.
    2. Generates modelled census estimates for ALL variables (fallback).
    3. Calls the Nomis API for the 5 live Census 2021 tables.
    4. Overwrites modelled values with real Nomis values where available.
    5. Returns the merged DataFrame and a list of real (non-modelled) columns.

    Church Street is correctly mapped as the most deprived ward (score 46.18).
    Westbourne is second most deprived (score 39.80).
    """

    # ------------------------------------------------------------------
    # Real IMD data - population-weighted ward averages from MHCLG
    # Rank 1 = most deprived in England; 6,904 = least deprived
    # ------------------------------------------------------------------
    imd = {
        "Ward": [
            # IMPORTANT: Church Street is the most deprived ward (score 46.18, rank 265)
            # Westbourne is second (score 39.80, rank 490)
            "Church Street", "Westbourne", "Queen's Park", "Harrow Road",
            "Pimlico South", "Vincent Square", "Pimlico North", "Maida Vale",
            "St James's", "Bayswater", "Little Venice", "West End",
            "Hyde Park", "Lancaster Gate", "Abbey Road",
            "Knightsbridge & Belgravia", "Marylebone", "Regent's Park"
        ],
        "IMD 2025 Score": [
            46.18, 39.80, 36.48, 34.60, 28.12, 24.53, 23.27, 21.75,
            20.91, 20.90, 21.34, 20.34, 18.05, 17.54, 17.55,
            13.96, 13.66, 13.46
        ],
        "IMD 2025 Rank": [
            265, 490, 661, 800, 1394, 1849, 2066, 2337,
            2499, 2501, 2411, 2611, 3103, 3234, 3233,
            4243, 4337, 4404
        ],
        "IMD 2019 Score": [
            41.45, 35.89, 32.40, 27.93, 21.63, 17.93, 19.26, 17.99,
            18.57, 23.18, 19.77, 14.60, 15.92, 15.79, 15.45,
            12.47, 12.17, 11.34
        ],
        "IMD 2019 Rank": [
            399, 658, 938, 1416, 2333, 3072, 2781, 3064,
            2933, 2061, 2666, 3972, 3589, 3629, 3719,
            4648, 4749, 5056
        ],
        # IMD Health Domain 2019 (standardised z-scores, negative = better than England avg)
        "Health Score 2019": [
            0.29, 0.29, 0.06, 0.00, -0.48, -0.58, -0.68, -1.04,
            -1.14, -0.73, -0.51, -1.26, -1.09, -0.92, -1.62,
            -1.72, -1.61, -1.93
        ],
    }
    df = pd.DataFrame(imd)

    # Score change and direction - flagged as directional only (not directly comparable)
    df["Score Change"]          = (df["IMD 2025 Score"] - df["IMD 2019 Score"]).round(2)
    df["Rank Change"]           = df["IMD 2019 Rank"] - df["IMD 2025 Rank"]
    df["Deprivation Direction"] = df["Score Change"].apply(
        lambda x: "Worsened" if x > 1 else ("Improved" if x < -1 else "Stable")
    )

    # National context bands (6,904 wards total in England, 2022 ONS boundaries)
    def nat_ctx(rank: int) -> str:
        if   rank <= TOTAL_WARDS_ENGLAND * 0.10: return "Top 10% most deprived"
        elif rank <= TOTAL_WARDS_ENGLAND * 0.20: return "Top 20% most deprived"
        elif rank <= TOTAL_WARDS_ENGLAND * 0.40: return "Top 40% most deprived"
        elif rank <= TOTAL_WARDS_ENGLAND * 0.80: return "Middle 40%"
        else:                                     return "Least deprived 20%"

    df["National Context"] = df["IMD 2025 Rank"].apply(nat_ctx)

    # Build full modelled estimates as fallback for ALL census variables
    modelled = _build_modelled_estimates(df)

    # Merge modelled estimates into the main dataframe
    out = df.merge(modelled.drop(columns=["Ward"]), left_index=True, right_index=True, how="left")

    # ------------------------------------------------------------------
    # Overwrite with real Nomis data where APIs succeed
    # ------------------------------------------------------------------
    real_census, api_status = load_nomis_census()
    real_cols = []   # track which columns now hold real (not modelled) data

    if not real_census.empty and len(real_census) >= 15:
        # Variables that come from each Nomis table
        nomis_vars = {
            "TS066": ["Employment Rate", "Unemployment %"],
            "TS007": ["Population"],
            "TS037": ["Good Health %", "Long-term Illness %"],
            "TS054": ["Owner Occupied %", "Social Rented %", "Private Rented %"],
            "TS067": ["No Qualifications %", "Degree Level %"],
        }
        for api, cols in nomis_vars.items():
            if api_status.get(api, False):
                for col in cols:
                    if col in real_census.columns:
                        # Merge real value into out, overwriting the modelled estimate
                        out = out.merge(
                            real_census[["Ward", col]].rename(columns={col: col + "_REAL"}),
                            on="Ward", how="left"
                        )
                        mask = out[col + "_REAL"].notna()
                        out.loc[mask, col] = out.loc[mask, col + "_REAL"]
                        out.drop(columns=[col + "_REAL"], inplace=True)
                        real_cols.append(col)

    # WCC quintile within Westminster only (Q1 = most deprived fifth)
    out["IMD Quintile"] = pd.qcut(
        out["IMD 2025 Score"], q=5,
        labels=["Q5 - Least deprived", "Q4", "Q3", "Q2", "Q1 - Most deprived"]
    )

    return out, real_cols


@st.cache_data
def load_age_profile():
    """
    Borough-level age distribution comparison - Westminster vs London.
    Source: ONS Census 2021 (TS007). Ward-level aggregated to borough.
    These are rounded published figures, not pulled via API.
    """
    bands = [
        "0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
        "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74",
        "75-79", "80-84", "85+"
    ]
    wcc    = [5.1, 4.2, 3.8, 4.5, 9.2, 10.8, 9.5, 8.1, 7.2, 6.3, 5.5,
              4.8, 3.9, 3.2, 2.8, 2.1, 1.5, 1.5]
    london = [6.2, 5.5, 4.9, 5.1, 7.9, 9.8, 9.2, 7.8, 6.7, 6.0, 5.3,
              4.5, 3.8, 3.1, 2.6, 1.9, 1.2, 1.5]
    return pd.DataFrame({"Age Band": bands, "Westminster %": wcc, "London %": london})


@st.cache_data
def load_industry_mix():
    """
    Employment by broad industry - Westminster vs London.
    Source: ONS Census 2021 (TS060). Borough-level published figures.
    """
    ind = [
        "Finance & Insurance", "Professional/Scientific", "Wholesale/Retail",
        "Accommodation & Food", "Public Admin", "Education",
        "Health & Social", "Info & Communication", "Arts & Entertainment",
        "Construction", "Other"
    ]
    return pd.DataFrame({
        "Industry":    ind,
        "Westminster": [14.2, 16.8, 9.1, 8.3, 6.2, 5.8, 7.4, 10.5, 4.2, 3.8, 13.7],
        "London":      [ 9.1, 12.3, 10.8, 7.9, 5.2, 7.1, 9.8,  8.4, 3.9, 4.6, 20.9],
    })


@st.cache_data
def load_geojson():
    """
    Load pre-converted GeoJSON for WCC 2022 ward boundaries.
    File: wcc_wards.geojson (converted from WCC_Wards2022_PBI.json TopoJSON).
    Must be committed to the same GitHub repo root as this Python file.
    Feature IDs match ward Label property (e.g. "Church Street").
    """
    geojson_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wcc_wards.geojson")
    if os.path.exists(geojson_path):
        with open(geojson_path) as f:
            return json.load(f)
    return None


# =============================================================================
# LOAD DATA + GLOBAL CONSTANTS
# =============================================================================
# Load with a spinner - Nomis calls can take a few seconds
with st.spinner("Loading data from Nomis..."):
    df, REAL_COLS = load_imd_and_census()

ages = load_age_profile()
ind  = load_industry_mix()
geo  = load_geojson()

WARDS = sorted(df["Ward"].tolist())

# Base feature set for regression/clustering - 4 predictors chosen to cover
# IMD's main domains while staying within the n/p > 10 guideline (18 wards / 4 predictors)
BASE_FEATS = ["Employment Rate", "No Qualifications %", "Social Rented %", "Overcrowding %"]

# Extended feature set used in Random Forest and correlation heatmap
ALL_CENSUS_FEATS = [
    "Employment Rate", "No Qualifications %", "Social Rented %",
    "Good Health %", "Overcrowding %", "Degree Level %",
    "Owner Occupied %", "Average Age", "Unemployment %", "Long-term Illness %"
]

# Only include features that actually exist in df (handles missing Nomis columns gracefully)
BASE_FEATS     = [f for f in BASE_FEATS     if f in df.columns]
ALL_CENSUS_FEATS = [f for f in ALL_CENSUS_FEATS if f in df.columns]


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## WCC Census & IMD")
    st.markdown("### Westminster City Council")
    st.markdown("---")

    page = st.radio("Navigate", [
        "Overview & IMD", "Deprivation Trends", "Demographics",
        "Housing & Tenure", "Economy & Labour", "Statistical Analysis",
        "Ward Map", "Data Sources & Quality", "How It's Built",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Filter wards**")
    sel_wards = st.multiselect(
        "Select wards (blank = all 18)",
        options=WARDS, default=[], placeholder="All 18 wards shown"
    )
    active = sel_wards if sel_wards else WARDS
    dff    = df[df["Ward"].isin(active)].copy()
    st.caption(f"Showing {len(dff)} of 18 wards")

    st.markdown("---")
    # Show which variables came from live Nomis vs modelled estimates
    if REAL_COLS:
        st.caption(
            f"**Live from Nomis ({len(REAL_COLS)}):** "
            + ", ".join(REAL_COLS)
        )
    else:
        st.caption("Nomis unavailable - all census variables are modelled estimates")

    st.caption(
        "\n\n**IMD:** [MHCLG 2025]"
        "(https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025)\n\n"
        "**Census:** [ONS 2021 via Nomis]"
        "(https://www.nomisweb.co.uk/sources/census_2021)\n\n"
        f"**England wards:** {TOTAL_WARDS_ENGLAND:,}"
    )


# =============================================================================
# PAGE HEADER
# =============================================================================
st.markdown("""
<div class="econ-header">
  <h1>Westminster City Council - Deprivation &amp; Census Dashboard</h1>
  <p>18 wards · IMD 2025 &amp; 2019 (MHCLG) · Census 2021 (ONS/Nomis live) · WCC 2022 ward boundaries</p>
</div>
""", unsafe_allow_html=True)

# Banner showing data freshness
if REAL_COLS:
    st.markdown(
        f'<div class="real-box">Live Census 2021 data loaded from Nomis for: '
        f'<b>{", ".join(REAL_COLS)}</b>. '
        f'Remaining variables use modelled estimates. Data refreshes hourly.</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="warn-box">Nomis API unavailable. All census variables shown are '
        'modelled estimates anchored to real IMD ward ranks. '
        'For real data, see the Data Sources &amp; Quality page.</div>',
        unsafe_allow_html=True
    )


# =============================================================================
# PAGE 1: OVERVIEW & IMD
# =============================================================================
if page == "Overview & IMD":

    most_dep  = df.loc[df["IMD 2025 Score"].idxmax(), "Ward"]
    least_dep = df.loc[df["IMD 2025 Score"].idxmin(), "Ward"]
    gap       = df["IMD 2025 Score"].max() - df["IMD 2025 Score"].min()
    worsened  = (df["Deprivation Direction"] == "Worsened").sum()

    st.markdown(
        f"""<div class="context-banner">Most deprived: <b>{most_dep}</b>
        (score {df['IMD 2025 Score'].max()}) &nbsp;|&nbsp;
        Least deprived: <b>{least_dep}</b> (score {df['IMD 2025 Score'].min()})
        &nbsp;|&nbsp; Borough gap: <b>{gap:.1f} pts</b>
        &nbsp;|&nbsp; Worsened since 2019: <b>{worsened}</b>
        &nbsp;|&nbsp; England total wards: <b>{TOTAL_WARDS_ENGLAND:,}</b></div>""",
        unsafe_allow_html=True
    )

    # KPI row
    # font-size 1.55rem + white-space:nowrap prevents 6-digit numbers wrapping
    k1, k2, k3, k4, k5 = st.columns(5)
    top10 = (dff["IMD 2025 Rank"] <= int(TOTAL_WARDS_ENGLAND * 0.1)).sum()
    with k1:
        st.markdown(f'<div class="kpi-card"><h2>{len(dff)}</h2><p>Wards shown</p></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card gold"><h2>{dff["IMD 2025 Score"].mean():.1f}</h2><p>Avg IMD 2025 Score</p></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card red"><h2>{dff["IMD 2025 Score"].max():.1f}</h2><p>Highest ({most_dep})</p></div>', unsafe_allow_html=True)
    with k4:
        # Formatted with commas; CSS white-space:nowrap keeps it on one line
        st.markdown(f'<div class="kpi-card"><h2>{dff["Population"].sum():,}</h2><p>Total population</p></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="kpi-card teal"><h2>{top10}</h2><p>Wards in top 10% nationally</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-label">Deprivation ranking</div>'
                    '<div class="section-title">IMD 2025 Score by ward - most to least deprived</div>',
                    unsafe_allow_html=True)

        s_df   = dff.sort_values("IMD 2025 Score", ascending=True)
        colors = [ECON_RED if s > 30 else WCC_GOLD if s > 20 else TEAL for s in s_df["IMD 2025 Score"]]

        fig = go.Figure(go.Bar(
            x=s_df["IMD 2025 Score"], y=s_df["Ward"], orientation="h",
            marker_color=colors,
            customdata=s_df[["IMD 2025 Rank", "National Context", "Score Change"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>Score: %{x:.2f}<br>"
                "Rank: %{customdata[0]:,} of 6,904<br>"
                "National band: %{customdata[1]}<br>"
                "Change 2019-2025: %{customdata[2]:+.2f}<extra></extra>"
            ),
        ))
        fig.add_vline(
            x=dff["IMD 2025 Score"].mean(), line_dash="dot", line_color=WCC_BLUE,
            line_width=1.5,
            annotation_text=f"Avg {dff['IMD 2025 Score'].mean():.1f}",
            annotation_position="top right",
            annotation_font=dict(size=9, color=WCC_BLUE)
        )
        fig = econ_h(fig, h=560,
                     src=f"MHCLG Index of Multiple Deprivation 2025 | {TOTAL_WARDS_ENGLAND:,} wards in England")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">National context</div>'
                    '<div class="section-title">How Westminster wards rank in England</div>',
                    unsafe_allow_html=True)

        ctx_order = [
            "Top 10% most deprived", "Top 20% most deprived",
            "Top 40% most deprived", "Middle 40%", "Least deprived 20%"
        ]
        ctx = (dff["National Context"].value_counts()
               .reindex(ctx_order, fill_value=0).reset_index())
        ctx.columns = ["Context", "Wards"]
        ctx = ctx[ctx["Wards"] > 0]

        fig2 = px.pie(
            ctx, names="Context", values="Wards", hole=0.45, color="Context",
            color_discrete_map={
                "Top 10% most deprived": ECON_RED,  "Top 20% most deprived": CORAL,
                "Top 40% most deprived": WCC_GOLD,  "Middle 40%": TEAL,
                "Least deprived 20%": WCC_BLUE,
            }
        )
        fig2.update_traces(textposition="outside", textfont_size=9)
        fig2 = econ(fig2, h=285, src="")
        fig2.update_layout(legend=dict(y=-0.4, font_size=9))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            f'<p style="font-size:0.71rem;color:#888">'
            f'Bands: top 10% = rank &le;{int(TOTAL_WARDS_ENGLAND*0.1):,} | '
            f'top 20% = &le;{int(TOTAL_WARDS_ENGLAND*0.2):,} | '
            f'top 40% = &le;{int(TOTAL_WARDS_ENGLAND*0.4):,} '
            f'(of {TOTAL_WARDS_ENGLAND:,} England wards)</p>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-label">Change 2019-2025</div>'
                    '<div class="section-title">Direction of deprivation</div>',
                    unsafe_allow_html=True)

        dir_df = df["Deprivation Direction"].value_counts().reset_index()
        dir_df.columns = ["Direction", "Wards"]
        fig3 = px.bar(dir_df, x="Direction", y="Wards", text="Wards", color="Direction",
                      color_discrete_map={"Worsened": ECON_RED, "Improved": TEAL, "Stable": WCC_GOLD})
        fig3.update_traces(textposition="outside")
        fig3 = econ(fig3, h=195, src="")
        fig3.update_layout(showlegend=False, margin=dict(t=15, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    # Correlation heatmap - all available census variables
    avail_cvars = [c for c in [
        "IMD 2025 Score", "Employment Rate", "No Qualifications %",
        "Social Rented %", "Good Health %", "Overcrowding %",
        "Degree Level %", "Unemployment %", "Long-term Illness %"
    ] if c in dff.columns]

    if len(avail_cvars) >= 3:
        st.markdown('<div class="section-label">Statistical relationships</div>'
                    '<div class="section-title">Correlation matrix - IMD score vs census variables (Pearson r)</div>',
                    unsafe_allow_html=True)
        src_note = "MHCLG IMD 2025; ONS Census 2021 via Nomis" if REAL_COLS else "MHCLG IMD 2025; ONS Census 2021 (modelled estimates)"
        corr_fig = px.imshow(
            dff[avail_cvars].corr(), text_auto=".2f", aspect="auto",
            color_continuous_scale=["#003087", "white", "#E3120B"], zmin=-1, zmax=1
        )
        corr_fig = econ(corr_fig, h=360,
                        subtitle="Pearson r. Red = strong positive; blue = strong negative.",
                        src=src_note)
        st.plotly_chart(corr_fig, use_container_width=True)


# =============================================================================
# PAGE 2: DEPRIVATION TRENDS
# =============================================================================
elif page == "Deprivation Trends":

    st.markdown(
        '<div class="warn-box">Comparability note: IMD 2019 and 2025 scores are not directly '
        "comparable - methodology, indicators and ward boundaries changed between editions. "
        "Treat score changes as directional signals only. "
        "Rank changes are more reliable for longitudinal comparison.</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">Slope chart</div>'
                    '<div class="section-title">IMD score 2019 vs 2025 by ward</div>',
                    unsafe_allow_html=True)
        fig_s = go.Figure()
        for _, row in dff.sort_values("IMD 2025 Score", ascending=False).iterrows():
            c = ECON_RED if row["Score Change"] > 1 else TEAL if row["Score Change"] < -1 else WCC_GOLD
            w = 2.5 if abs(row["Score Change"]) > 3 else 1.5
            fig_s.add_trace(go.Scatter(
                x=[2019, 2025], y=[row["IMD 2019 Score"], row["IMD 2025 Score"]],
                mode="lines+markers+text", name=row["Ward"],
                line=dict(color=c, width=w), marker=dict(size=6),
                text=["", row["Ward"]], textposition="middle right", textfont=dict(size=8),
                hovertemplate=(f"<b>{row['Ward']}</b><br>2019: {row['IMD 2019 Score']}<br>"
                               f"2025: {row['IMD 2025 Score']}<br>Change: {row['Score Change']:+.2f}<extra></extra>"),
            ))
        fig_s.update_layout(xaxis=dict(tickvals=[2019, 2025], showgrid=False, zeroline=False, showline=False),
                            yaxis_title="IMD Score", showlegend=False)
        fig_s = econ(fig_s, h=560, src="MHCLG IMD 2025 and 2019")
        fig_s.update_layout(margin=dict(r=130))
        st.plotly_chart(fig_s, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">Score change</div>'
                    '<div class="section-title">IMD score change 2019 to 2025</div>',
                    unsafe_allow_html=True)
        cd = dff[["Ward", "Score Change", "Deprivation Direction"]].sort_values("Score Change", ascending=False)
        fig_c = px.bar(cd, x="Score Change", y="Ward", orientation="h",
                       color="Deprivation Direction", text="Score Change",
                       color_discrete_map={"Worsened": ECON_RED, "Improved": TEAL, "Stable": WCC_GOLD})
        fig_c.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
        fig_c.add_vline(x=0, line_color="#999", line_width=1)
        fig_c = econ_h(fig_c, h=560, src="MHCLG IMD 2025 and 2019")
        fig_c.update_layout(legend=dict(y=-0.1))
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown('<div class="section-label">Full data table</div>'
                '<div class="section-title">Ward-level IMD detail</div>',
                unsafe_allow_html=True)
    cols = ["Ward", "IMD 2019 Score", "IMD 2019 Rank", "IMD 2025 Score", "IMD 2025 Rank",
            "Score Change", "Rank Change", "Deprivation Direction", "National Context"]
    st.dataframe(df[cols].sort_values("IMD 2025 Score", ascending=False).reset_index(drop=True),
                 use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 3: DEMOGRAPHICS
# =============================================================================
elif page == "Demographics":

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">Age</div>'
                    '<div class="section-title">Westminster vs London - age distribution</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=ages["Age Band"], x=ages["Westminster %"],
                             name="Westminster", orientation="h", marker_color=WCC_BLUE))
        fig.add_trace(go.Bar(y=ages["Age Band"], x=ages["London %"],
                             name="London", orientation="h", marker_color=ECON_RED, opacity=0.55))
        fig.update_layout(barmode="group", xaxis_title="% of population", legend=dict(x=0.55, y=0.05))
        fig = econ_h(fig, h=420, src="ONS Census 2021 (TS007)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">Ethnicity</div>'
                    '<div class="section-title">Estimated ethnicity mix by ward</div>',
                    unsafe_allow_html=True)
        eth_cols = [c for c in ["White %", "Asian %", "Black %", "Mixed %"] if c in dff.columns]
        if eth_cols:
            eth = dff[["Ward"] + eth_cols].melt(id_vars="Ward", var_name="Ethnicity", value_name="%")
            fig2 = px.bar(eth, x="Ward", y="%", color="Ethnicity", barmode="stack",
                          color_discrete_map={"White %": WCC_BLUE, "Asian %": TEAL, "Black %": CORAL, "Mixed %": SAGE})
            fig2 = econ(fig2, h=420, src="ONS Census 2021 (TS021, modelled estimates)", rotated=True)
            fig2.update_layout(xaxis_tickangle=-45, legend=dict(y=-0.3))
            st.plotly_chart(fig2, use_container_width=True)

    hlth_cols = [c for c in ["Good Health %", "Long-term Illness %"] if c in dff.columns]
    if hlth_cols:
        st.markdown('<div class="section-label">Health</div>'
                    '<div class="section-title">Health outcomes by ward - sorted by deprivation (highest first)</div>',
                    unsafe_allow_html=True)
        hlth = dff.sort_values("IMD 2025 Score", ascending=False)
        fig3 = go.Figure()
        if "Good Health %"       in hlth.columns: fig3.add_trace(go.Bar(x=hlth["Ward"], y=hlth["Good Health %"],       name="Good Health %",       marker_color=TEAL))
        if "Long-term Illness %" in hlth.columns: fig3.add_trace(go.Bar(x=hlth["Ward"], y=hlth["Long-term Illness %"], name="Long-term Illness %", marker_color=ECON_RED))
        fig3.update_layout(barmode="group", yaxis_title="%", xaxis_tickangle=-45)
        src_h = "ONS Census 2021 via Nomis (TS037)" if "Good Health %" in REAL_COLS else "ONS Census 2021 (TS037, modelled estimates)"
        fig3 = econ(fig3, h=340, src=src_h, rotated=True)
        st.plotly_chart(fig3, use_container_width=True)


# =============================================================================
# PAGE 4: HOUSING & TENURE
# =============================================================================
elif page == "Housing & Tenure":

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">Tenure</div>'
                    '<div class="section-title">Tenure mix by ward - sorted by deprivation (highest first)</div>',
                    unsafe_allow_html=True)
        ten_avail = [c for c in ["Owner Occupied %", "Social Rented %", "Private Rented %"] if c in dff.columns]
        if ten_avail:
            ten = dff.sort_values("IMD 2025 Score", ascending=False)
            ten_m = ten[["Ward"] + ten_avail].melt(id_vars="Ward", var_name="Tenure", value_name="%")
            fig = px.bar(ten_m, x="%", y="Ward", orientation="h", barmode="stack", color="Tenure",
                         color_discrete_map={"Owner Occupied %": WCC_BLUE, "Social Rented %": ECON_RED, "Private Rented %": TEAL})
            src_t = "ONS Census 2021 via Nomis (TS054)" if "Social Rented %" in REAL_COLS else "ONS Census 2021 (TS054, modelled estimates)"
            fig = econ_h(fig, h=520, src=src_t)
            fig.update_layout(legend=dict(y=-0.12))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Social Rented %" in dff.columns and "Overcrowding %" in dff.columns:
            st.markdown('<div class="section-label">Overcrowding</div>'
                        '<div class="section-title">Social renting vs overcrowding</div>',
                        unsafe_allow_html=True)
            fig2 = px.scatter(dff, x="Social Rented %", y="Overcrowding %",
                              color="IMD 2025 Score", hover_name="Ward", size="Population", trendline="ols",
                              color_continuous_scale=["#84B59F", "#C8A84B", "#E3120B"])
            fig2 = econ(fig2, h=270, src="ONS Census 2021; MHCLG IMD 2025")
            fig2.update_layout(coloraxis_colorbar=dict(title="IMD 2025"))
            st.plotly_chart(fig2, use_container_width=True)

        if "Owner Occupied %" in dff.columns and "Median Rooms" in dff.columns:
            st.markdown('<div class="section-label">Room size</div>'
                        '<div class="section-title">Owner occupancy vs median rooms</div>',
                        unsafe_allow_html=True)
            fig3 = px.scatter(dff, x="Owner Occupied %", y="Median Rooms", hover_name="Ward",
                              trendline="ols", color="IMD 2025 Score",
                              color_continuous_scale=["#84B59F", "#E3120B"])
            fig3 = econ(fig3, h=265, src="ONS Census 2021 (TS050, TS054, modelled estimates)")
            st.plotly_chart(fig3, use_container_width=True)


# =============================================================================
# PAGE 5: ECONOMY & LABOUR
# =============================================================================
elif page == "Economy & Labour":

    col1, col2 = st.columns([3, 2])
    with col1:
        if "Employment Rate" in dff.columns:
            st.markdown('<div class="section-label">Employment</div>'
                        '<div class="section-title">Employment rate by ward</div>',
                        unsafe_allow_html=True)
            emp = dff.sort_values("Employment Rate")
            fig = px.bar(emp, x="Ward", y="Employment Rate", color="IMD 2025 Score", text="Employment Rate",
                         color_continuous_scale=["#003087", "#C8A84B", "#E3120B"])
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            src_e = "ONS Census 2021 via Nomis (TS066)" if "Employment Rate" in REAL_COLS else "ONS Census 2021 (TS066, modelled)"
            fig = econ(fig, h=380, src=src_e, rotated=True)
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">Industry</div>'
                    '<div class="section-title">Westminster vs London sector mix</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Westminster", x=ind["Industry"], y=ind["Westminster"], marker_color=WCC_BLUE))
        fig2.add_trace(go.Bar(name="London",      x=ind["Industry"], y=ind["London"],      marker_color=ECON_RED, opacity=0.65))
        fig2.update_layout(barmode="group", yaxis_title="%", xaxis_tickangle=-45)
        fig2 = econ(fig2, h=380, src="ONS Census 2021 (TS060, WCC & London borough-level)", rotated=True)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "Degree Level %" in dff.columns and "Employment Rate" in dff.columns:
            st.markdown('<div class="section-label">Education link</div>'
                        '<div class="section-title">Degree attainment vs employment rate</div>',
                        unsafe_allow_html=True)
            fig3 = px.scatter(dff, x="Degree Level %", y="Employment Rate", hover_name="Ward",
                              trendline="ols", color="IMD 2025 Score",
                              color_continuous_scale=["#003087", "#E3120B"],
                              size="Population", size_max=30)
            fig3 = econ(fig3, h=340, src="ONS Census 2021 (TS066, TS067)")
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if "Unemployment %" in dff.columns:
            st.markdown('<div class="section-label">Labour market</div>'
                        '<div class="section-title">Unemployment rate vs IMD score</div>',
                        unsafe_allow_html=True)
            fig4 = px.scatter(dff, x="IMD 2025 Score", y="Unemployment %", hover_name="Ward",
                              trendline="ols", color="Deprivation Direction",
                              color_discrete_map={"Worsened": ECON_RED, "Improved": TEAL, "Stable": WCC_GOLD},
                              size="Population", size_max=30)
            fig4 = econ(fig4, h=340, src="ONS Census 2021; MHCLG IMD 2025")
            st.plotly_chart(fig4, use_container_width=True)


# =============================================================================
# PAGE 6: STATISTICAL ANALYSIS
# =============================================================================
elif page == "Statistical Analysis":

    if len(BASE_FEATS) < 2:
        st.error("Not enough census variables available for statistical analysis. Check Nomis connection.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Linear Regression", "Random Forest", "Clustering", "Model Validation",
    ])

    # =========================================================================
    # TAB 1: LINEAR REGRESSION (OLS)
    # =========================================================================
    with tab1:
        st.markdown("### Linear Regression - predicting IMD 2025 Score")
        st.markdown("""<div class="method-box">
        <b>What is OLS regression?</b><br>
        Ordinary Least Squares (OLS) fits a straight line through the data by minimising the sum
        of squared residuals - the vertical distances between each observed and predicted value.
        Coefficients tell you: for a 1-unit increase in predictor X, how much does the IMD score
        change <i>holding all other predictors constant</i>?<br><br>
        <b>R-squared:</b> Proportion of variance in IMD scores explained by the model.
        R-squared = 0.85 means 85% of ward-to-ward variation is captured.
        Adjusted R-squared penalises extra predictors that don't genuinely improve fit.<br><br>
        <b>P-values:</b> Probability of observing a coefficient this large by chance if the true
        effect is zero. P &lt; 0.05 = conventional significance threshold.
        With n=18 wards, all tests are low-powered - non-significance may reflect small n,
        not absence of a real relationship.<br><br>
        <b>95% Confidence intervals:</b> The range within which the true coefficient would fall
        95% of the time under repeated sampling. Wide intervals are expected at n=18.<br><br>
        <b>Rule of thumb (n=18):</b> Use no more than 4 predictors (n/p &gt; 10). Adding more
        risks overfitting and inflates standard errors.
        </div>""", unsafe_allow_html=True)

        sel = st.multiselect("Predictor variables (X)", ALL_CENSUS_FEATS, default=BASE_FEATS[:4])

        if len(sel) >= 1:
            X_raw = df[sel].dropna(axis=0).values
            y_raw = df.loc[df[sel].notna().all(axis=1), "IMD 2025 Score"].values
            ols   = sm.OLS(y_raw, sm.add_constant(X_raw)).fit()
            y_hat = ols.predict(sm.add_constant(X_raw))
            resid = y_raw - y_hat
            dw    = durbin_watson(resid)
            sw_s, sw_p = shapiro(resid)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R-squared",     f"{ols.rsquared:.3f}")
            m2.metric("Adj. R-sq",     f"{ols.rsquared_adj:.3f}")
            m3.metric("RMSE",          f"{np.sqrt(mean_squared_error(y_raw, y_hat)):.2f}")
            m4.metric("F p-value",     f"{ols.f_pvalue:.4f}")

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_raw, y=y_hat, mode="markers+text",
                    text=df.loc[df[sel].notna().all(axis=1), "Ward"].values,
                    textposition="top center", textfont=dict(size=8),
                    marker=dict(color=[ECON_RED if e > 0 else TEAL for e in resid],
                                size=9, line=dict(width=1, color="white")),
                    hovertemplate="<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
                ))
                fig.add_shape(type="line", x0=y_raw.min(), y0=y_raw.min(),
                              x1=y_raw.max(), y1=y_raw.max(),
                              line=dict(color=ECON_GREY, dash="dash"))
                fig = econ(fig, title="Actual vs Predicted",
                           subtitle="Points above diagonal = model underestimates deprivation", src="", h=380)
                fig.update_layout(xaxis_title="Actual IMD", yaxis_title="Predicted IMD")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                coef_df = pd.DataFrame({
                    "Feature": sel, "Coef": ols.params[1:],
                    "CI_low": ols.conf_int()[1:, 0], "CI_high": ols.conf_int()[1:, 1],
                    "p_value": ols.pvalues[1:],
                }).sort_values("Coef")
                fig2 = go.Figure()
                for _, r in coef_df.iterrows():
                    c  = ECON_RED if r["Coef"] > 0 else TEAL
                    op = 1.0 if r["p_value"] < 0.05 else 0.35
                    fig2.add_trace(go.Scatter(x=[r["CI_low"], r["CI_high"]], y=[r["Feature"], r["Feature"]],
                                              mode="lines", line=dict(color=c, width=2), opacity=op, showlegend=False))
                    fig2.add_trace(go.Scatter(x=[r["Coef"]], y=[r["Feature"]], mode="markers",
                                              marker=dict(color=c, size=10), opacity=op, showlegend=False,
                                              hovertemplate=f"<b>{r['Feature']}</b><br>Coef: {r['Coef']:.3f}<br>95% CI: [{r['CI_low']:.3f}, {r['CI_high']:.3f}]<br>p={r['p_value']:.3f}<extra></extra>"))
                fig2.add_vline(x=0, line_color=ECON_GREY, line_width=1.5)
                fig2 = econ(fig2, title="Coefficients + 95% CI",
                            subtitle="Faded = p > 0.05 (not significant). CI crossing zero = not significant.", src="", h=380)
                fig2.update_layout(xaxis_title="Coefficient (effect per 1-unit increase)")
                st.plotly_chart(fig2, use_container_width=True)

            ct = coef_df.copy()
            ct["95% CI"]    = ct.apply(lambda r: f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}]", axis=1)
            ct["p-value"]   = ct["p_value"].apply(lambda p: f"{p:.4f}{'*' if p < 0.05 else ''}")
            ct["Direction"] = ct["Coef"].apply(lambda c: "Increases deprivation" if c > 0 else "Reduces deprivation")
            st.dataframe(ct[["Feature", "Coef", "95% CI", "p-value", "Direction"]].rename(columns={"Coef": "Coefficient"}),
                         use_container_width=True, hide_index=True)

            st.markdown(f"""<div class="method-box">
            <b>Diagnostics</b><br>
            <span class="stat-pill">Durbin-Watson = {dw:.3f}</span>
            {'No autocorrelation (DW near 2)' if 1.5 < dw < 2.5 else 'Possible autocorrelation - note DW is designed for time series; Moran\'s I is more appropriate for cross-sectional ward data'}<br><br>
            <span class="stat-pill">Shapiro-Wilk W={sw_s:.4f}, p={sw_p:.4f}</span>
            {'Residuals plausibly normal (p > 0.05)' if sw_p > 0.05 else 'Residuals may not be normal - CIs may be unreliable'}<br><br>
            <span class="stat-pill">F-stat={ols.fvalue:.2f}, p={ols.f_pvalue:.4f}</span>
            {'Model statistically significant overall' if ols.f_pvalue < 0.05 else 'Model not significant at p < 0.05'}
            </div>""", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: RANDOM FOREST
    # =========================================================================
    with tab2:
        st.markdown("### Random Forest - Feature Importance")
        st.markdown("""<div class="method-box">
        <b>What is Random Forest?</b><br>
        An ensemble of decision trees, each trained on a bootstrap sample of wards using a
        random subset of features. Final predictions are averages across all trees.
        The bootstrapping and random feature selection reduce overfitting compared to a single tree.<br><br>
        <b>Feature importance (MDI - Mean Decrease Impurity):</b><br>
        Measures how often each feature splits a node and how much it reduces variance at each split,
        averaged across all trees. Higher = more important for predicting IMD score.
        MDI can be biased towards continuous/high-cardinality features; permutation importance
        (not shown here) is a more robust alternative.<br><br>
        <b>OOB R-squared (Out-of-Bag):</b><br>
        Each tree trains on a bootstrap sample, which excludes roughly 37% of wards by chance.
        Those excluded wards are the "out-of-bag" set for that tree.
        OOB R-squared is computed by predicting each ward only from trees that never trained on it.
        This gives a near-unbiased generalisation estimate without needing a held-out test set -
        ideal at n=18 where we cannot afford to hold out a test split.<br><br>
        <b>Key check:</b> A large gap between Train R-squared and OOB R-squared signals overfitting.
        Try reducing max depth or increasing n_estimators if this occurs.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        n_t = c1.slider("Number of trees", 10, 300, 100, 10)
        m_d = c2.slider("Max tree depth",  1,  10,   5)

        rf_feats_all = ALL_CENSUS_FEATS + (["Median Rooms"] if "Median Rooms" in df.columns else [])
        # Deduplicate and only use available columns
        rf_feats = list(dict.fromkeys([f for f in rf_feats_all if f in df.columns]))

        X_rf = df[rf_feats].fillna(df[rf_feats].median()).values
        y_rf = df["IMD 2025 Score"].values

        rf = RandomForestRegressor(n_estimators=n_t, max_depth=m_d, random_state=42, oob_score=True)
        rf.fit(X_rf, y_rf)
        rf_pred = rf.predict(X_rf)

        m1, m2, m3 = st.columns(3)
        m1.metric("Train R-sq",  f"{r2_score(y_rf, rf_pred):.3f}")
        m2.metric("OOB R-sq",    f"{rf.oob_score_:.3f}",
                  help="Each ward predicted only by trees that never trained on it - unbiased estimate")
        m3.metric("Trees", str(n_t))

        col1, col2 = st.columns(2)
        with col1:
            imp = pd.DataFrame({"Feature": rf_feats, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=True)
            fig = go.Figure(go.Bar(
                x=imp["Importance"], y=imp["Feature"], orientation="h",
                marker_color=[ECON_RED if v > 0.15 else WCC_GOLD if v > 0.08 else TEAL for v in imp["Importance"]],
                text=imp["Importance"].apply(lambda x: f"{x:.3f}"), textposition="outside"
            ))
            fig = econ_h(fig, title="Feature Importance (MDI)",
                         subtitle="Higher = more important. Red > 0.15, gold > 0.08.",
                         h=440, src="scikit-learn RandomForestRegressor")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=y_rf, y=rf_pred, mode="markers+text",
                text=df["Ward"], textposition="top center", textfont=dict(size=8),
                marker=dict(color=WCC_BLUE, size=9, line=dict(width=1, color="white")),
                hovertemplate="<b>%{text}</b><br>Actual: %{x:.2f}<br>RF: %{y:.2f}<extra></extra>"
            ))
            fig2.add_shape(type="line", x0=y_rf.min(), y0=y_rf.min(), x1=y_rf.max(), y1=y_rf.max(),
                           line=dict(color=ECON_GREY, dash="dash"))
            fig2 = econ(fig2, title="Actual vs RF Predicted",
                        subtitle=f"OOB R-sq = {rf.oob_score_:.3f} (unbiased)", src="", h=440)
            fig2.update_layout(xaxis_title="Actual IMD", yaxis_title="RF Predicted")
            st.plotly_chart(fig2, use_container_width=True)

        if rf.oob_score_ < r2_score(y_rf, rf_pred) - 0.15:
            st.markdown(
                f'<div class="warn-box">Train R-sq ({r2_score(y_rf, rf_pred):.3f}) vs OOB R-sq ({rf.oob_score_:.3f}) - '
                "large gap suggests overfitting. Reduce max depth or increase trees.</div>",
                unsafe_allow_html=True
            )

    # =========================================================================
    # TAB 3: CLUSTERING
    # =========================================================================
    with tab3:
        st.markdown("### K-Means Clustering - Ward Typologies")
        st.markdown("""<div class="method-box">
        <b>What is K-Means clustering?</b><br>
        An unsupervised algorithm that partitions 18 wards into k groups by minimising
        within-cluster sum of squares (WCSS). Starting with k random centroids, it assigns each
        ward to the nearest centroid, recalculates centroids, and repeats until stable.
        Running n_init=10 uses 10 random restarts to avoid local minima.<br><br>
        <b>Why StandardScaler is essential before K-Means:</b><br>
        K-Means uses Euclidean distance. A variable in hundreds (population) would dominate
        a percentage variable (0-100) without scaling. StandardScaler transforms each variable
        to mean=0, standard deviation=1, so all variables contribute equally to the distance
        calculation. Cluster profiles shown use original unscaled values for interpretability.<br><br>
        <b>Choosing k:</b><br>
        No single "correct" k exists. Use the Validation tab's elbow plot, silhouette score
        and Davies-Bouldin index together, then choose based on policy interpretability.
        With n=18, k=3 or k=4 gives the most actionable ward typologies.<br><br>
        <b>Limitations:</b><br>
        K-Means assumes spherical clusters of equal density. With only 18 wards, any cluster
        with fewer than 4 members should be interpreted cautiously.
        </div>""", unsafe_allow_html=True)

        cl_options = [f for f in [
            "IMD 2025 Score", "Employment Rate", "Degree Level %",
            "Social Rented %", "Overcrowding %", "Good Health %",
            "Average Age", "Unemployment %"
        ] if f in df.columns]

        c1, c2 = st.columns(2)
        n_k = c1.slider("Number of clusters", 2, 6, 3)
        cl_feats = c2.multiselect(
            "Variables to cluster on", cl_options,
            default=[f for f in ["IMD 2025 Score", "Employment Rate", "Social Rented %", "Overcrowding %"] if f in cl_options]
        )

        if len(cl_feats) >= 2:
            X_cl  = df[cl_feats].fillna(df[cl_feats].median()).values
            X_sc  = StandardScaler().fit_transform(X_cl)
            km    = KMeans(n_clusters=n_k, random_state=42, n_init=10)
            labels = km.fit_predict(X_sc)
            df_cl = df.copy()
            df_cl["Cluster"] = "Type " + (labels + 1).astype(str)

            sil = silhouette_score(X_sc, labels)
            dbi = davies_bouldin_score(X_sc, labels)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(
                    df_cl, x=cl_feats[0],
                    y=cl_feats[1] if len(cl_feats) > 1 else "IMD 2025 Score",
                    color="Cluster", hover_name="Ward",
                    size="Population", text="Ward",
                    color_discrete_sequence=[WCC_BLUE, ECON_RED, TEAL, SAGE, WCC_GOLD, "#6D2E46"]
                )
                fig.update_traces(textposition="top center", textfont_size=8,
                                  selector=dict(mode="markers+text"))
                fig = econ(fig, title=f"Ward Clusters (k={n_k})",
                           subtitle=f"Silhouette = {sil:.3f} | Davies-Bouldin = {dbi:.3f}",
                           src="ONS Census 2021; MHCLG IMD 2025", h=420)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Cluster profiles - mean values (original scale)**")
                # Deduplicate groupby columns - prevents Arrow serialisation error
                # when user selects "IMD 2025 Score" as a clustering variable
                # (it would otherwise appear twice in the column list)
                groupby_cols = list(dict.fromkeys(cl_feats + ["Population", "IMD 2025 Score"]))
                cluster_summary = (
                    df_cl.groupby("Cluster")[groupby_cols]
                    .mean().round(1).reset_index()
                )
                st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

                st.markdown("**Ward assignments**")
                wc = df_cl[["Ward", "Cluster", "IMD 2025 Score", "Population"]].sort_values(
                    ["Cluster", "IMD 2025 Score"], ascending=[True, False])
                st.dataframe(wc, use_container_width=True, height=240, hide_index=True)

            m1, m2 = st.columns(2)
            m1.metric("Silhouette Score", f"{sil:.3f}",
                      help=">0.5 strong | 0.25-0.5 weak | <0.25 possibly spurious")
            m2.metric("Davies-Bouldin Index", f"{dbi:.3f}",
                      help="Lower = better separated clusters")

    # =========================================================================
    # TAB 4: MODEL VALIDATION
    # =========================================================================
    with tab4:
        st.markdown("### Model Validation - How Robust Are These Results?")
        st.markdown("""<div class="alert-box">
        n=18 wards throughout. All statistical tests are low-powered with small samples.
        Confidence intervals are wide. Treat outputs as exploratory, triangulated with
        qualitative evidence and domain knowledge.
        A non-significant result does not mean no real relationship exists.
        </div>""", unsafe_allow_html=True)

        vt1, vt2, vt3, vt4 = st.tabs([
            "Regression Diagnostics", "Cross-Validation",
            "AUC and DeLong Test", "Cluster Validation",
        ])

        # Pre-fit base regression for use across sub-tabs
        X_base_raw = df[BASE_FEATS].fillna(df[BASE_FEATS].median()).values
        y_base     = df["IMD 2025 Score"].values
        ols_base   = sm.OLS(y_base, sm.add_constant(X_base_raw)).fit()
        resid_base = y_base - ols_base.predict(sm.add_constant(X_base_raw))

        # -----------------------------------------------------------------
        with vt1:
            st.markdown("#### Regression Diagnostics - base model")
            st.markdown(f"""<div class="method-box">
            <b>Durbin-Watson (DW):</b> Tests for autocorrelation in residuals.
            Values near 2.0 = no autocorrelation; below 1.5 = positive autocorrelation;
            above 2.5 = negative autocorrelation. DW was designed for time series data.
            For cross-sectional ward data, Moran's I (which tests whether geographically adjacent
            wards have more similar residuals than random) is theoretically more appropriate.
            DW is used here as a practical proxy, but its results should be interpreted cautiously.<br><br>

            <b>Shapiro-Wilk test:</b> Tests whether residuals follow a normal distribution.
            Normally distributed residuals are an assumption of OLS inference (p-values, CIs).
            P > 0.05 = cannot reject normality (plausibly normal).
            P &lt; 0.05 = evidence of non-normality, which may invalidate confidence intervals.
            With n=18, this test has low statistical power to detect mild non-normality.<br><br>

            <b>Q-Q plot:</b> Visual normality check. Points close to the diagonal line =
            normally distributed residuals. An S-curve pattern suggests skewness;
            heavy tails at both ends suggest kurtosis (more extreme outliers than normal).
            The Q-Q plot is often more informative than the Shapiro-Wilk test alone at small n.<br><br>

            <b>Residuals vs Fitted:</b> Checks for homoscedasticity (constant residual variance).
            Ideal pattern = random scatter around zero across all fitted values.
            A funnel shape (variance increasing with fitted value) suggests heteroscedasticity,
            which makes OLS standard errors and p-values unreliable.<br><br>

            <b>VIF (Variance Inflation Factor):</b> Detects multicollinearity between predictors.
            When predictors are correlated, their coefficients become unstable and hard to interpret.
            VIF = 1 = no inflation. VIF &gt; 5 = moderate concern. VIF &gt; 10 = serious problem.
            High VIF means coefficients are unreliable even if R-squared looks good.<br><br>

            <b>Cook's Distance:</b> Identifies which wards exert disproportionate influence on
            the regression line. A ward with high Cook's D is "pulling" the fitted line towards it.
            Threshold = 4/n = {4/len(df):.3f} for n=18. High-leverage wards warrant closer inspection -
            they may be genuine outliers or data quality issues.
            </div>""", unsafe_allow_html=True)

            dw_b = durbin_watson(resid_base)
            sw_b, sw_pb = shapiro(resid_base)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Durbin-Watson", f"{dw_b:.3f}")
            c2.metric("Shapiro-Wilk p", f"{sw_pb:.4f}")
            c3.metric("F-statistic",  f"{ols_base.fvalue:.2f}")
            c4.metric("F p-value",    f"{ols_base.f_pvalue:.4f}")

            col1, col2 = st.columns(2)
            with col1:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(
                    x=ols_base.fittedvalues, y=resid_base, mode="markers+text",
                    text=df["Ward"], textposition="top center", textfont=dict(size=8),
                    marker=dict(color=[ECON_RED if abs(r) > 2*resid_base.std() else WCC_BLUE for r in resid_base],
                                size=9, line=dict(width=1, color="white")),
                    hovertemplate="<b>%{text}</b><br>Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>"
                ))
                fig_r.add_hline(y=0, line_color=ECON_GREY, line_dash="dash")
                fig_r.add_hline(y= 2*resid_base.std(), line_color=WCC_GOLD, line_dash="dot", line_width=1, annotation_text="+ 2 sd", annotation_position="right")
                fig_r.add_hline(y=-2*resid_base.std(), line_color=WCC_GOLD, line_dash="dot", line_width=1, annotation_text="- 2 sd", annotation_position="right")
                fig_r = econ(fig_r, title="Residuals vs Fitted",
                             subtitle="Red = |residual| > 2 sd. Random scatter = homoscedastic.", src="", h=360)
                fig_r.update_layout(xaxis_title="Fitted values", yaxis_title="Residuals")
                st.plotly_chart(fig_r, use_container_width=True)

            with col2:
                (osm, osr) = stats.probplot(resid_base, dist="norm")
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(x=osm[0], y=osm[1], mode="markers",
                                            marker=dict(color=WCC_BLUE, size=9), name="Residuals"))
                fig_qq.add_trace(go.Scatter(x=osm[0], y=osr[1] + osr[0]*np.array(osm[0]),
                                            mode="lines", line=dict(color=WCC_GOLD, dash="dash"), name="Normal ref"))
                fig_qq = econ(fig_qq, title="Q-Q Plot of Residuals",
                              subtitle=f"Shapiro-Wilk W={sw_b:.4f}, p={sw_pb:.4f}. "
                                       f"{'Plausibly normal' if sw_pb > 0.05 else 'Departure from normality - CIs may be unreliable'}",
                              src="", h=360)
                fig_qq.update_layout(xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles", showlegend=False)
                st.plotly_chart(fig_qq, use_container_width=True)

            st.markdown("#### VIF - multicollinearity check")
            vif_df = pd.DataFrame({
                "Feature": BASE_FEATS,
                "VIF":     [variance_inflation_factor(X_base_raw, i) for i in range(X_base_raw.shape[1])],
            })
            vif_df["Assessment"] = vif_df["VIF"].apply(
                lambda v: "OK (< 5)" if v < 5 else ("Moderate (5-10)" if v < 10 else "High (> 10)"))
            st.dataframe(vif_df, use_container_width=True, hide_index=True)

            st.markdown("#### Cook's Distance - influential wards")
            cooks, _ = ols_base.get_influence().cooks_distance
            thresh   = 4 / len(df)
            fig_ck = go.Figure(go.Bar(
                x=df["Ward"], y=cooks,
                marker_color=[ECON_RED if c > thresh else TEAL for c in cooks],
                hovertemplate="<b>%{x}</b><br>Cook's D: %{y:.4f}<extra></extra>"
            ))
            fig_ck.add_hline(y=thresh, line_dash="dash", line_color=WCC_GOLD,
                             annotation_text=f"Threshold 4/n={thresh:.3f}", annotation_position="right")
            fig_ck = econ(fig_ck, title="Cook's Distance",
                          subtitle="Above threshold = high leverage on regression coefficients", src="", h=300, rotated=True)
            fig_ck.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_ck, use_container_width=True)

        # -----------------------------------------------------------------
        with vt2:
            st.markdown("#### Cross-Validation - estimating out-of-sample performance")
            st.markdown("""<div class="method-box">
            <b>Why cross-validate?</b><br>
            Training and testing on the same data gives an overly optimistic performance estimate.
            Cross-validation holds out subsets of data for testing, giving a fairer view of how
            the model would generalise to new wards it has never seen.<br><br>

            <b>Leave-One-Out CV (LOO-CV):</b><br>
            The most statistically efficient approach at n=18. Each ward is held out once as
            the sole test observation while the model trains on the remaining 17.
            LOO R-squared = mean across all 18 folds. Individual fold scores are noisy (test n=1)
            but the average provides a reliable generalisation estimate.<br><br>

            <b>5-Fold CV:</b><br>
            Data split into 5 folds of ~3-4 wards each. Each fold is the test set once.
            With test folds of only 3-4 wards at n=18, fold-to-fold variance is high.
            Mean and standard deviation across folds are both reported - the standard deviation
            is as important as the mean at this sample size.<br><br>

            <b>Negative CV R-squared:</b><br>
            A negative R-squared means the model predicts worse than simply guessing the mean
            for every ward. This can happen when a test fold contains an unusual ward that
            the model trained on very different data cannot generalise to.
            Negative individual fold scores are not uncommon at n=18 and do not mean the
            model is useless - examine the LOO per-ward plot to see which wards are hardest to predict.
            </div>""", unsafe_allow_html=True)

            lr_sk = LinearRegression()
            rf_cv = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            loo   = LeaveOneOut()
            kf5   = KFold(n_splits=5, shuffle=True, random_state=42)

            loo_lr = cross_val_score(lr_sk, X_base_raw, y_base, cv=loo, scoring="r2")
            loo_rf = cross_val_score(rf_cv, X_base_raw, y_base, cv=loo, scoring="r2")
            kf_lr  = cross_val_score(lr_sk, X_base_raw, y_base, cv=kf5, scoring="r2")
            kf_rf  = cross_val_score(rf_cv, X_base_raw, y_base, cv=kf5, scoring="r2")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("LOO R-sq (Linear)", f"{loo_lr.mean():.3f}", delta=f"sd {loo_lr.std():.3f}")
            m2.metric("LOO R-sq (RF)",     f"{loo_rf.mean():.3f}", delta=f"sd {loo_rf.std():.3f}")
            m3.metric("5-Fold (Linear)",   f"{kf_lr.mean():.3f}",  delta=f"sd {kf_lr.std():.3f}")
            m4.metric("5-Fold (RF)",       f"{kf_rf.mean():.3f}",  delta=f"sd {kf_rf.std():.3f}")

            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(x=list(range(1, 6)), y=kf_lr, name="Linear", marker_color=WCC_BLUE))
            fig_cv.add_trace(go.Bar(x=list(range(1, 6)), y=kf_rf, name="Random Forest", marker_color=CORAL))
            fig_cv.add_hline(y=0, line_color=ECON_GREY, line_width=1)
            fig_cv = econ(fig_cv, title="5-Fold CV - R-squared per fold",
                          subtitle="High variance between folds is expected at n=18",
                          src="scikit-learn cross_val_score", h=310)
            fig_cv.update_layout(xaxis_title="Fold", yaxis_title="R-squared", barmode="group")
            st.plotly_chart(fig_cv, use_container_width=True)

            fig_loo = go.Figure()
            fig_loo.add_trace(go.Scatter(x=df["Ward"], y=loo_lr, mode="markers+lines",
                                         name="Linear", marker_color=WCC_BLUE, marker_size=8))
            fig_loo.add_trace(go.Scatter(x=df["Ward"], y=loo_rf, mode="markers+lines",
                                         name="RF", marker_color=CORAL, marker_size=8))
            fig_loo.add_hline(y=0, line_color=ECON_GREY, line_dash="dash")
            fig_loo = econ(fig_loo, title="LOO CV - R-squared when each ward is held out",
                           subtitle="Negative = model cannot predict this ward from the other 17",
                           src="", h=340, rotated=True)
            fig_loo.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_loo, use_container_width=True)

        # -----------------------------------------------------------------
        with vt3:
            st.markdown("#### AUC, ROC Curve and DeLong-style Comparison")
            st.markdown("""<div class="method-box">
            <b>Converting to a classification problem:</b><br>
            AUC and ROC are metrics for binary classifiers. We dichotomise IMD Score at the
            median: wards above median = "high deprivation" (1), wards at or below = 0.
            Two logistic regression models are compared: Model A (4 predictors) vs Model B (2).<br><br>

            <b>AUC (Area Under the ROC Curve):</b><br>
            The ROC curve plots True Positive Rate (sensitivity) against False Positive Rate
            (1 - specificity) at every possible classification threshold.
            AUC = the probability that the model ranks a randomly chosen high-deprivation ward
            above a randomly chosen low-deprivation ward.
            AUC = 0.5 is no better than random. AUC = 1.0 is perfect classification.<br><br>

            <b>5-Fold Stratified AUC:</b><br>
            Stratified k-fold ensures each fold contains approximately equal proportions of
            high/low deprivation wards - important with a balanced binary outcome at small n.
            Mean and standard deviation of AUC across 5 folds provides an out-of-sample estimate.
            At n=18, standard deviations of 0.1-0.2 in AUC are common and not alarming.<br><br>

            <b>DeLong test (bootstrap approximation):</b><br>
            The formal DeLong (1988) test compares two correlated AUC estimates from the same
            held-out test set. Here we use 1,000 bootstrap resamples as an approximation because
            the formal version requires paired predictions from a held-out test set, which at n=18
            we cannot reliably create. The bootstrap distribution of (AUC_A - AUC_B) is used to
            compute a 95% CI and p-value. If the CI excludes zero, Model A has a significantly
            higher AUC. At n=18, the CI will almost always include zero - this reflects
            statistical power constraints, not equivalence of the models in practice.
            Any comparison should be confirmed with additional data before informing policy.
            </div>""", unsafe_allow_html=True)

            med   = df["IMD 2025 Score"].median()
            y_bin = (df["IMD 2025 Score"] > med).astype(int)

            fa = BASE_FEATS
            fb = [f for f in ["Employment Rate", "Degree Level %"] if f in df.columns]

            if len(fa) >= 2 and len(fb) >= 2:
                Xa  = StandardScaler().fit_transform(df[fa].fillna(df[fa].median()).values)
                Xb  = StandardScaler().fit_transform(df[fb].fillna(df[fb].median()).values)
                lrA = LogisticRegression(max_iter=1000, random_state=42)
                lrB = LogisticRegression(max_iter=1000, random_state=42)
                lrA.fit(Xa, y_bin); pA = lrA.predict_proba(Xa)[:, 1]
                lrB.fit(Xb, y_bin); pB = lrB.predict_proba(Xb)[:, 1]
                aA = roc_auc_score(y_bin, pA); aB = roc_auc_score(y_bin, pB)

                skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_A = cross_val_score(LogisticRegression(max_iter=1000, random_state=42), Xa, y_bin, cv=skf, scoring="roc_auc")
                cv_B = cross_val_score(LogisticRegression(max_iter=1000, random_state=42), Xb, y_bin, cv=skf, scoring="roc_auc")

                np.random.seed(99)
                diffs = []
                for _ in range(1000):
                    idx = np.random.choice(len(y_bin), len(y_bin), replace=True)
                    yb  = y_bin.iloc[idx]
                    if len(np.unique(yb)) < 2: continue
                    try:
                        diffs.append(roc_auc_score(yb, pA[idx]) - roc_auc_score(yb, pB[idx]))
                    except Exception:
                        pass
                diffs  = np.array(diffs)
                ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
                z_stat = diffs.mean() / (diffs.std() + 1e-9)
                p_dl   = 2 * (1 - stats.norm.cdf(abs(z_stat)))

                col1, col2 = st.columns(2)
                with col1:
                    fpA, tpA, _ = roc_curve(y_bin, pA); fpB, tpB, _ = roc_curve(y_bin, pB)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpA, y=tpA, mode="lines", name=f"Model A (4 vars) AUC={aA:.3f}", line=dict(color=WCC_BLUE, width=2.5)))
                    fig_roc.add_trace(go.Scatter(x=fpB, y=tpB, mode="lines", name=f"Model B (2 vars) AUC={aB:.3f}", line=dict(color=ECON_RED, width=2.5, dash="dash")))
                    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random (0.5)", line=dict(color=ECON_GREY, dash="dot")))
                    fig_roc = econ(fig_roc, title="ROC Curves - Models A and B",
                                   subtitle=f"Binary target: IMD score > {med:.1f} (median)",
                                   src="scikit-learn LogisticRegression", h=420)
                    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", legend=dict(y=-0.3))
                    st.plotly_chart(fig_roc, use_container_width=True)

                with col2:
                    m1, m2 = st.columns(2)
                    m1.metric("AUC - Model A", f"{aA:.3f}"); m2.metric("AUC - Model B", f"{aB:.3f}")
                    m1.metric("5-Fold AUC A",  f"{cv_A.mean():.3f}", delta=f"sd {cv_A.std():.3f}")
                    m2.metric("5-Fold AUC B",  f"{cv_B.mean():.3f}", delta=f"sd {cv_B.std():.3f}")
                    fig_b = go.Figure(go.Histogram(x=diffs, nbinsx=40, marker_color=WCC_BLUE, opacity=0.75))
                    fig_b.add_vline(x=0, line_color=WCC_GOLD, line_dash="dash")
                    fig_b.add_vrect(x0=ci_lo, x1=ci_hi, fillcolor=WCC_GOLD, opacity=0.15,
                                    annotation_text="95% CI", annotation_position="top left")
                    fig_b = econ(fig_b, title="Bootstrap AUC difference (A - B)",
                                 subtitle=f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}] | p = {p_dl:.3f}",
                                 src="1,000 bootstrap resamples", h=260)
                    st.plotly_chart(fig_b, use_container_width=True)

                st.markdown(f"""<div class="method-box">
                <b>DeLong-style result:</b>
                <span class="stat-pill">AUC diff = {aA-aB:.3f}</span>
                <span class="stat-pill">95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]</span>
                <span class="stat-pill">p = {p_dl:.3f}</span><br><br>
                {'95% CI excludes zero - Model A has a significantly higher AUC.' if (ci_lo > 0 or ci_hi < 0) else
                 'CI includes zero - no significant AUC difference. Expected at n=18: sample too small to reliably detect moderate AUC differences. Does not mean the models are equivalent.'}<br><br>
                <b>Caveat:</b> Bootstrap resampling approximates the formal DeLong (1988) method,
                which requires paired predictions from a held-out test set.
                Any AUC comparison should be confirmed with additional data before informing policy.
                </div>""", unsafe_allow_html=True)

        # -----------------------------------------------------------------
        with vt4:
            st.markdown("#### Cluster Validation - choosing the right k")
            st.markdown("""<div class="method-box">
            <b>Elbow plot (WCSS):</b><br>
            Within-Cluster Sum of Squares decreases as k increases - adding more clusters
            always explains more variance. Look for the "elbow" where the rate of decrease
            sharply flattens. Beyond this k, extra clusters give diminishing returns.
            The elbow is a visual heuristic and can be ambiguous; use alongside the other metrics.<br><br>

            <b>Silhouette score:</b><br>
            For each ward, silhouette = (b - a) / max(a, b), where:
            - a = mean distance to other wards in the same cluster
            - b = mean distance to wards in the nearest other cluster
            Values near +1 = well-separated, clearly assigned cluster.
            Near 0 = overlapping clusters. Negative = possibly assigned to wrong cluster.
            The overall score is the mean across all 18 wards.
            Guidance: &gt;0.5 = strong structure; 0.25-0.5 = weak; &lt;0.25 = possibly spurious.<br><br>

            <b>Davies-Bouldin Index (DBI):</b><br>
            For each cluster pair, DBI = (within-cluster scatter A + within-cluster scatter B) /
            distance between centroids A and B. DBI is the average of the worst such ratio across all
            cluster pairs. Lower = better (clusters are compact and well-separated from each other).
            DBI penalises clusters that are close together even if individually tight.
            Silhouette and DBI can disagree - use both to triangulate the best k value.
            </div>""", unsafe_allow_html=True)

            cl_b = [f for f in ["IMD 2025 Score", "Employment Rate", "Social Rented %", "Overcrowding %"] if f in df.columns]
            X_v  = StandardScaler().fit_transform(df[cl_b].fillna(df[cl_b].median()).values)

            k_range = range(2, 9)
            wcss, sils, dbis = [], [], []
            for k in k_range:
                km_v = KMeans(n_clusters=k, random_state=42, n_init=10)
                lv   = km_v.fit_predict(X_v)
                wcss.append(km_v.inertia_)
                sils.append(silhouette_score(X_v, lv))
                dbis.append(davies_bouldin_score(X_v, lv))

            best_sil = list(k_range)[int(np.argmax(sils))]
            best_dbi = list(k_range)[int(np.argmin(dbis))]

            col1, col2, col3 = st.columns(3)
            with col1:
                fig_e = go.Figure(go.Scatter(x=list(k_range), y=wcss, mode="lines+markers",
                                             marker=dict(color=WCC_BLUE, size=8), line=dict(color=WCC_BLUE, width=2.5)))
                fig_e = econ(fig_e, title="Elbow Plot (WCSS)", subtitle="Look for sharp flattening", src="", h=270)
                fig_e.update_layout(xaxis_title="k", yaxis_title="WCSS")
                st.plotly_chart(fig_e, use_container_width=True)

            with col2:
                fig_s = go.Figure(go.Bar(x=list(k_range), y=sils,
                                         marker_color=[ECON_RED if k == best_sil else TEAL for k in k_range],
                                         text=[f"{s:.3f}" for s in sils], textposition="outside"))
                fig_s = econ(fig_s, title="Silhouette Score", subtitle=f"Best k = {best_sil}", src="", h=270)
                fig_s.update_layout(xaxis_title="k", yaxis_title="Silhouette")
                st.plotly_chart(fig_s, use_container_width=True)

            with col3:
                fig_d = go.Figure(go.Bar(x=list(k_range), y=dbis,
                                         marker_color=[ECON_RED if k == best_dbi else TEAL for k in k_range],
                                         text=[f"{d:.3f}" for d in dbis], textposition="outside"))
                fig_d = econ(fig_d, title="Davies-Bouldin Index", subtitle=f"Best k = {best_dbi}", src="", h=270)
                fig_d.update_layout(xaxis_title="k", yaxis_title="DBI")
                st.plotly_chart(fig_d, use_container_width=True)

            st.markdown(
                f'<div class="insight-box">Silhouette suggests k={best_sil}; '
                f"Davies-Bouldin suggests k={best_dbi}. "
                f"{'These agree - k=' + str(best_sil) + ' is robust.' if best_sil == best_dbi else 'These disagree. Try both k values and choose based on which ward typology is most useful for your policy purpose.'} "
                "With n=18, k=3 or k=4 typically produces the most interpretable groups.</div>",
                unsafe_allow_html=True
            )


# =============================================================================
# PAGE 7: WARD MAP
# Choropleth using converted WCC 2022 GeoJSON ward boundaries
# =============================================================================
elif page == "Ward Map":

    st.markdown("## Ward Map - IMD 2025 by Geography")
    st.markdown("""<div class="context-banner">
    Boundaries: WCC 2022 ward review. GeoJSON converted from TopoJSON (WCC_Wards2022_PBI.json).
    Commit wcc_wards.geojson alongside nomis_dashboard.py to your GitHub repo for deployment.
    </div>""", unsafe_allow_html=True)

    if geo is None:
        st.error(
            "wcc_wards.geojson not found in the same directory as nomis_dashboard.py. "
            "Download it from the dashboard build outputs and commit it to your GitHub repo."
        )
    else:
        map_var = st.selectbox("Colour map by", [
            "IMD 2025 Score", "IMD 2019 Score", "Score Change",
            "Employment Rate", "Unemployment %", "Social Rented %",
            "Overcrowding %", "Degree Level %", "Good Health %",
        ])
        # Positive variables: flip scale so red still means "worse"
        pos_vars = ["Good Health %", "Employment Rate", "Degree Level %"]
        cscale   = "Blues_r" if map_var in pos_vars else "Reds"

        map_df = df[["Ward", map_var]].copy()

        fig_map = px.choropleth_map(
            map_df, geojson=geo, locations="Ward", color=map_var,
            featureidkey="id",
            center={"lat": 51.513, "lon": -0.145}, zoom=12,
            map_style="carto-positron", color_continuous_scale=cscale,
            hover_data={"Ward": True, map_var: ":.2f"}, opacity=0.75,
        )
        fig_map.update_layout(
            height=600, margin=dict(l=0, r=0, t=30, b=10),
            coloraxis_colorbar=dict(title=map_var, len=0.6, thickness=14, tickfont=dict(size=10))
        )
        st.plotly_chart(fig_map, use_container_width=True)

        col1, col2 = st.columns([3, 2])
        with col2:
            st.markdown(f"**{map_var} - 18 wards ranked**")
            map_table = df[["Ward", map_var, "National Context"]].sort_values(map_var, ascending=False).reset_index(drop=True)
            st.dataframe(map_table, use_container_width=True, hide_index=True, height=400)
        with col1:
            sorted_map = df.sort_values(map_var, ascending=True)
            bar_colors = [
                ECON_RED if v > sorted_map[map_var].quantile(0.75)
                else WCC_GOLD if v > sorted_map[map_var].median() else TEAL
                for v in sorted_map[map_var]
            ]
            fig_bar = go.Figure(go.Bar(x=sorted_map[map_var], y=sorted_map["Ward"], orientation="h",
                                       marker_color=bar_colors, hovertemplate="<b>%{y}</b><br>" + map_var + ": %{x:.2f}<extra></extra>"))
            fig_bar = econ_h(fig_bar, title=map_var,
                             subtitle="Red = top quartile | Gold = above median | Teal = below median",
                             h=500, src="MHCLG IMD 2025; ONS Census 2021")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("""<div class="warn-box">
        Ward boundary file: converted from WCC_Wards2022_PBI.json (TopoJSON, WCC GIS team).
        Coordinates in WGS84. For high-precision spatial analysis use the original shapefile.
        </div>""", unsafe_allow_html=True)


# =============================================================================
# PAGE 8: DATA SOURCES & QUALITY
# =============================================================================
elif page == "Data Sources & Quality":

    st.markdown("## Data Sources, Methodology and Quality Assurance")
    ds1, ds2, ds3 = st.tabs(["IMD Data", "Census 2021 / Nomis", "Quality and Limitations"])

    with ds1:
        st.markdown("### Index of Multiple Deprivation (IMD)")
        st.markdown(f"""
**Primary source:** MHCLG
- [IMD 2025](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025)
- [IMD 2025 Technical Report](https://www.gov.uk/government/publications/english-indices-of-deprivation-2025-technical-report)
- [IMD 2019](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)

National total: **{TOTAL_WARDS_ENGLAND:,} wards in England** (ONS 2022 boundaries) used for percentile bands.
        """)
        st.dataframe(pd.DataFrame({
            "Domain": ["Income", "Employment", "Education, Skills & Training",
                       "Health Deprivation & Disability", "Crime",
                       "Barriers to Housing & Services", "Living Environment"],
            "Weight": ["22.5%", "22.5%", "13.5%", "13.5%", "9.3%", "9.3%", "9.3%"],
            "What it covers": [
                "Low-income households; benefit claimants; tax credit recipients",
                "Involuntary exclusion from paid work",
                "Lack of attainment and skills in the local population",
                "Risk of premature death; impaired quality of life through ill health",
                "Violence, burglary, theft, criminal damage rates",
                "Physical and financial barriers to decent housing and services",
                "Air quality, housing condition, road accident rates"
            ],
        }), use_container_width=True, hide_index=True)
        st.markdown("""<div class="warn-box">
        IMD 2025 and 2019 are NOT directly comparable. Methodology, indicator reference years
        and ward boundaries all changed between editions. Score changes = directional signals only.
        Rank changes are more reliable for longitudinal comparison.
        </div>""", unsafe_allow_html=True)

    with ds2:
        st.markdown("### Census 2021 via Nomis Live API")
        st.markdown("""
**Source:** ONS Census 2021, England and Wales. Reference date: 21 March 2021.
- [All Census 2021 tables via Nomis](https://www.nomisweb.co.uk/sources/census_2021)
- [Census Table Finder](https://www.nomisweb.co.uk/census/2021/data_finder)
        """)

        st.markdown("**Five live API tables loaded by this dashboard:**")
        st.dataframe(pd.DataFrame({
            "Table": ["TS066", "TS007", "TS037", "TS054", "TS067"],
            "Nomis ID": ["NM_2083_1", "NM_2027_1", "NM_2055_1", "NM_2072_1", "NM_2084_1"],
            "Variables extracted": [
                "Employment Rate, Unemployment %",
                "Total population per ward",
                "Good Health %, Long-term Illness % (proxy)",
                "Owner Occupied %, Social Rented %, Private Rented %",
                "No Qualifications %, Degree Level %"
            ],
            "Measure": ["Percentage (20301)", "Count (20100)", "Percentage (20301)",
                        "Percentage (20301)", "Count (20100 - % calculated manually)"],
        }), use_container_width=True, hide_index=True)

        st.markdown("**Geography parameters used in all API calls:**")
        st.code(
            f"geography=TYPE298 (2022 wards)\n"
            f"geography_filter=PARENT:{WESTMINSTER_PARENT} (Westminster LA)\n"
            f"Data refreshes every 60 minutes (Streamlit cache TTL=3600)",
            language="text"
        )

        st.markdown("**Example API call (TS066 employment):**")
        st.code(
            f"https://www.nomisweb.co.uk/api/v01/dataset/NM_2083_1.data.csv"
            f"?date=latest"
            f"&geography=TYPE298&geography_filter=PARENT:{WESTMINSTER_PARENT}"
            f"&c2021_eastat_20=1001,1006,1011"
            f"&measures=20301",
            language="text"
        )

        if REAL_COLS:
            st.markdown(f'<div class="real-box">Currently live from Nomis: <b>{", ".join(REAL_COLS)}</b></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">Nomis unavailable - all census variables are modelled estimates.</div>',
                        unsafe_allow_html=True)

        st.markdown("""<div class="alert-box">
        Variables NOT covered by the 5 API tables (ethnicity, overcrowding, average age,
        long-term illness, median rooms) use modelled estimates anchored to real IMD ward ranks.
        Always check the sidebar for which columns are live vs modelled before using in reports.
        </div>""", unsafe_allow_html=True)

    with ds3:
        st.markdown("### Quality Issues and Analytical Limitations")
        qa = [
            ("Census: response rate variation", "ONS Quality Guide 2021",
             "Response rates varied across wards. Church Street and Harrow Road fell below the national average (~89%). Lower response rates increase coverage bias risk, particularly for younger males, private renters and recent migrants. Data for these groups in high non-response wards should be treated with added caution.",
             "warn"),
            ("Census: Statistical Disclosure Control", "ONS Census 2021 Methodology",
             "ONS applied Targeted Record Swapping (households with unusual characteristics may be swapped with a nearby similar household) and Cell Key Perturbation (small counts adjusted by +/-1-2). Ward-level counts for small population subgroups may not exactly equal the true figure.",
             "warn"),
            ("Census: COVID-19 reference date", "ONS Census 2021",
             "Reference date: 21 March 2021. Employment, health and commuting data reflect pandemic conditions. Economic activity in particular should be treated with care as it may not represent long-term structural patterns.",
             "warn"),
            ("IMD: temporal comparability", "MHCLG Technical Report, Section 4",
             "IMD 2025 uses data reference years from 2019 to 2023 depending on the indicator. Scores are not directly comparable across editions. Use rank changes rather than score changes for longitudinal comparison.",
             "warn"),
            ("IMD: LSOA to ward aggregation", "MHCLG / ONS",
             "IMD is calculated at LSOA level (approx. 1,500 population). Ward scores are population-weighted averages of LSOAs. This masks within-ward variation: a ward with a moderate average score may contain both very deprived and very affluent LSOAs.",
             "alert"),
            ("IMD: relative measure", "MHCLG Technical Report, Section 2",
             "IMD measures deprivation relative to other areas in England. An improving rank does not necessarily mean absolute conditions improved - it may mean the area fell behind other areas less rapidly.",
             "warn"),
            ("Analysis: ecological fallacy", "Statistical methodology",
             "Ward-level relationships do not necessarily hold for individuals within those wards. Do not infer individual behaviour from these aggregate statistics.",
             "alert"),
            ("Analysis: spatial autocorrelation", "Statistical methodology",
             "Adjacent wards are likely more similar than distant ones. OLS regression assumes independent observations. Moran's I is the appropriate test for spatial autocorrelation in residuals; Durbin-Watson is used here as a proxy but was designed for time-series data.",
             "warn"),
        ]
        for title, src, desc, box in qa:
            st.markdown(f'<div class="{box}-box"><b>{title}</b> <span style="color:#888;font-size:0.79rem">({src})</span><br><br>{desc}</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 9: HOW IT'S BUILT
# =============================================================================
elif page == "How It's Built":
    st.markdown("## How This Dashboard Works")
    t1, t2, t3, t4 = st.tabs(["The Stack", "Pandas", "Plotly", "Deploy"])

    with t1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### GitHub")
            st.markdown("- Version control\n- Free public hosting\n- Auto-triggers Streamlit redeploy on push")
        with c2:
            st.markdown("### Python")
            st.markdown("- `requests` - live Nomis API calls\n- `pandas` - data wrangling\n- `plotly` - interactive charts\n- `scikit-learn` - ML models\n- `statsmodels` - OLS regression\n- `scipy` - statistical tests\n- `streamlit` - web app")
        with c3:
            st.markdown("### Streamlit")
            st.markdown("- Every widget reruns the script top to bottom\n- `@st.cache_data(ttl=3600)` - live Nomis data refreshes hourly\n- Free deployment on Streamlit Community Cloud")

    with t2:
        st.code("""
import pandas as pd, requests, io

# Live Nomis API call - no API key needed for Census 2021 public data
url = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_2083_1.data.csv"
    "?date=latest"
    "&geography=TYPE298&geography_filter=PARENT:1946157124"
    "&c2021_eastat_20=1001,1006,1011"   # In employment, Unemployed, Inactive
    "&measures=20301"                    # Percentage
)
resp = requests.get(url, timeout=30)
df   = pd.read_csv(io.StringIO(resp.text))
# Filter for employment rows and sum percentage per ward
df_emp = df[df["c2021_eastat_20_name"].str.contains("In employment", case=False)]
df_emp.groupby("geography_name")["obs_value"].sum()
        """, language="python")

    with t3:
        st.code("""
import plotly.graph_objects as go

fig = go.Figure(go.Bar(
    x=df["IMD 2025 Score"], y=df["Ward"], orientation="h",
    marker_color=["red" if s > 30 else "gold" for s in df["IMD 2025 Score"]]
))
fig.add_vline(x=df["IMD 2025 Score"].mean(), line_dash="dot",
              annotation_text="Borough average")
st.plotly_chart(fig, use_container_width=True)
        """, language="python")

    with t4:
        st.code("""
# requirements.txt - in GitHub repo root
streamlit
pandas
numpy
plotly
scikit-learn
statsmodels
scipy
requests

# Files in repo root:
# nomis_dashboard.py
# requirements.txt
# wcc_wards.geojson    <-- download from the build output

# Deploy: share.streamlit.io > New App > select repo + nomis_dashboard.py
        """, language="bash")
        st.success("Live at a shareable URL. Anyone can view it without installing anything.")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(
    f"Westminster City Council - 18 wards (2022 boundaries) - "
    f"[IMD 2025 (MHCLG)](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025) - "
    f"[Census 2021 (ONS/Nomis)](https://www.nomisweb.co.uk/sources/census_2021) - "
    f"England wards: {TOTAL_WARDS_ENGLAND:,} - "
    f"Census data: {'live from Nomis (' + str(len(REAL_COLS)) + ' vars)' if REAL_COLS else 'modelled estimates'} - "
    f"Built with Python and Streamlit"
)
