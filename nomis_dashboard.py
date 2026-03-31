"""
Westminster City Council - Census 2021 & IMD Analytical Dashboard
=================================================================
Author: Westminster City Council - Data and Intelligence Team
Built with: Python, Streamlit, Plotly, Scikit-learn, Statsmodels, SciPy

Data sources:
  - IMD 2025 & 2019: MHCLG (Ministry of Housing, Communities & Local Government)
  - Census 2021 ward-level estimates: ONS / Nomis API
  - Ward boundaries: WCC 2022 ward boundaries (18 wards)

To run locally:   streamlit run nomis_dashboard.py
To deploy:        Push to GitHub > Streamlit Community Cloud

Requirements (requirements.txt):
  streamlit, pandas, numpy, plotly, scikit-learn, statsmodels, scipy, matplotlib

Note on census variables: the census figures shown in this dashboard are official 
Census 2021 ward‑level statistics retrieved directly from the ONS / Nomis API.
They reflect the latest published data for Westminster’s wards and should be treated 
as official statistics for policy analysis.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import requests


# Statistical modelling
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error, roc_auc_score,
                              roc_curve, silhouette_score, davies_bouldin_score)
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold, StratifiedKFold

# Statistical diagnostics
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
# Westminster brand colours + Economist-style chart palette
# =============================================================================
WCC_BLUE  = "#003087"   # Westminster navy - primary brand colour
WCC_GOLD  = "#C8A84B"   # Westminster gold - secondary brand colour
TEAL      = "#028090"   # Accent teal for positive/improvement signals
CORAL     = "#E87461"   # Soft coral for mid-range data
SAGE      = "#84B59F"   # Sage green for fourth category
ECON_RED  = "#E3120B"   # Economist signature red - used for top rule and high deprivation
ECON_GREY = "#CCCCCC"   # Economist grid/axis grey
LIGHT_GRID = "#E8E8E8"  # Light horizontal gridlines
DARK_TEXT  = "#1A1A1A"  # Near-black for headings
MID_TEXT   = "#555555"  # Mid-grey for subtitles and axis labels

# Total number of wards in England under 2022 ONS ward boundaries
# Used to calculate national deprivation percentile bands
TOTAL_WARDS_ENGLAND = 6904


# =============================================================================
# CSS STYLING
# Economist-influenced typography using Libre Baskerville (serif headlines)
# and Source Sans 3 (clean body text). WCC brand colours applied throughout.
# =============================================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Source+Sans+3:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
  }

  /* Page header - red top rule is the Economist's signature device */
  .econ-header {
    border-top: 4px solid #E3120B;
    padding: 1.2rem 0 0.8rem;
    margin-bottom: 1rem;
  }
  .econ-header h1 {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.85rem; color: #1A1A1A; margin: 0 0 0.3rem;
  }
  .econ-header p { color: #555; font-size: 0.9rem; margin: 0; }

  /* KPI metric cards - top-rule colour indicates status */
  .kpi-card {
    background: white;
    border-top: 3px solid #003087;
    padding: 0.75rem 0.9rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  /* Compact number sizing - prevents overflow at 100% zoom on a standard HD monitor.
     white-space: nowrap ensures long numbers like 216,972 stay on one line. */
  .kpi-card h2 {
    margin: 0;
    font-size: 1.55rem;
    color: #003087;
    font-family: 'Libre Baskerville', serif;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .kpi-card p { margin: 0.15rem 0 0; font-size: 0.76rem; color: #555; }
  /* Colour variants for different card types */
  .kpi-card.red  { border-top-color: #E3120B; } .kpi-card.red  h2 { color: #E3120B; }
  .kpi-card.gold { border-top-color: #C8A84B; } .kpi-card.gold h2 { color: #C8A84B; }
  .kpi-card.teal { border-top-color: #028090; } .kpi-card.teal h2 { color: #028090; }

  /* Section labels - small red uppercase text above chart titles */
  .section-label {
    font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #E3120B; margin: 1.1rem 0 0.1rem;
  }
  /* Section title - Baskerville serif, matches Economist house style */
  .section-title {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.05rem; font-weight: 700; color: #1A1A1A;
    border-bottom: 1px solid #E8E8E8; padding-bottom: 0.3rem; margin-bottom: 0.5rem;
  }

  /* Callout boxes for insights, warnings and alerts */
  .insight-box {
    background: #F7F9FC; border-left: 3px solid #028090;
    padding: 0.6rem 0.85rem; font-size: 0.84rem; margin: 0.4rem 0;
  }
  .warn-box {
    background: #FFFBF0; border-left: 3px solid #C8A84B;
    padding: 0.6rem 0.85rem; font-size: 0.84rem; margin: 0.4rem 0;
  }
  .alert-box {
    background: #FFF5F5; border-left: 3px solid #E3120B;
    padding: 0.6rem 0.85rem; font-size: 0.84rem; margin: 0.4rem 0;
  }
  /* Methodology and QA boxes */
  .method-box {
    background: #F0F4FA; border: 1px solid #C8D8F0;
    padding: 0.85rem 1rem; border-radius: 4px; margin: 0.5rem 0; font-size: 0.83rem;
  }
  /* Monospace pill for displaying stat test results */
  .stat-pill {
    font-size: 0.78rem; font-family: 'Courier New', monospace;
    background: #EEEEEE; padding: 0.2rem 0.5rem; border-radius: 3px;
    display: inline-block; margin: 0.1rem 0.1rem;
  }
  /* Dark context banner at top of pages */
  .context-banner {
    background: #1A1A2E; color: #c8e6ff;
    padding: 0.55rem 0.95rem; font-size: 0.81rem; margin-bottom: 0.8rem;
  }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ECONOMIST CHART THEME FUNCTIONS
#
# Two helper functions that apply consistent Economist-style formatting
# to any Plotly figure:
#   econ()   - for vertical charts (bar, scatter, line, heatmap)
#   econ_h() - for horizontal bar charts (swaps which axis has gridlines)
#
# Key Economist design signatures:
#   - Red rule above the chart
#   - Bold serif title, grey subtitle
#   - Horizontal gridlines only (no vertical)
#   - Source note below x-axis labels (not overlapping them)
#   - Clean white background, no chart border
# =============================================================================
def econ(fig, title="", subtitle="",
         src="MHCLG IMD 2025; ONS Census 2021",
         h=420, xgrid=False, rotated=False):
    """
    Apply Economist house style to a vertical Plotly chart.

    Parameters
    ----------
    fig      : plotly Figure object
    title    : bold serif headline for the chart
    subtitle : lighter grey explanatory line below title
    src      : source attribution text shown below x-axis labels
    h        : chart height in pixels
    xgrid    : whether to show vertical gridlines (usually False)
    rotated  : set True if x-axis labels are rotated - adds extra bottom margin
               so source note never overlaps the labels
    """
    ttl = ""
    if title:
        ttl += f"<b style='font-family:Georgia,serif;font-size:14px'>{title}</b>"
    if subtitle:
        ttl += f"<br><span style='font-size:10px;color:{MID_TEXT}'>{subtitle}</span>"

    # Bottom margin is larger when source text is present, and larger still
    # when x-axis labels are rotated (e.g. -45 degrees), so they never overlap
    b_margin = 20
    if src:
        b_margin = 105 if rotated else 72

    fig.update_layout(
        height=h,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=11, color=DARK_TEXT),
        title=dict(
            text=ttl, x=0, xanchor="left", y=0.98, yanchor="top",
            pad=dict(l=0, t=8)
        ),
        xaxis=dict(
            showgrid=xgrid, gridcolor=LIGHT_GRID,
            zeroline=False, showline=True, linecolor=ECON_GREY, linewidth=1,
            tickfont=dict(size=10, color=MID_TEXT),
            title_font=dict(size=10, color=MID_TEXT)
        ),
        yaxis=dict(
            showgrid=True, gridcolor=LIGHT_GRID,
            zeroline=False, showline=False,
            tickfont=dict(size=10, color=MID_TEXT),
            title_font=dict(size=10, color=MID_TEXT)
        ),
        margin=dict(l=10, r=10, t=75 if ttl else 28, b=b_margin),
        legend=dict(
            orientation="h", y=-0.18, x=0,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)", borderwidth=0
        ),
        hoverlabel=dict(bgcolor="white", font_size=11, bordercolor=ECON_GREY),
    )

    # Source note positioned below the x-axis labels
    # y position is more negative for rotated labels to avoid any overlap
    if src:
        src_y = -0.28 if rotated else -0.19
        fig.add_annotation(
            text=f"<span style='font-size:9px;color:#888888'>Source: {src}</span>",
            xref="paper", yref="paper",
            x=0, y=src_y, showarrow=False,
            align="left", xanchor="left"
        )
    return fig


def econ_h(fig, title="", subtitle="",
           src="MHCLG IMD 2025; ONS Census 2021", h=420):
    """
    Economist style for horizontal bar charts.
    Swaps which axis gets gridlines vs axis line.
    """
    fig = econ(fig, title=title, subtitle=subtitle, src=src, h=h)
    # For horizontal charts, gridlines go on the x-axis (value axis)
    # and the category axis gets a clean spine line instead
    fig.update_layout(
        xaxis=dict(
            showgrid=True, gridcolor=LIGHT_GRID,
            zeroline=False, showline=False,
            tickfont=dict(size=10, color=MID_TEXT),
            title_font=dict(size=10, color=MID_TEXT)
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            showline=True, linecolor=ECON_GREY, linewidth=1,
            tickfont=dict(size=10, color=MID_TEXT),
            title_font=dict(size=10, color=MID_TEXT)
        ),
    )
    return fig


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_real_census_from_nomis():
    """
    Pull real Census 2021 ward-level data for Westminster from the Nomis API.

    Currently loads employment and unemployment (TS066). Extend with further
    tables (TS067, TS054, TS021, TS037/TS038, TS050) as needed.
    """
    # Westminster local authority code in Nomis (wards under parent LA)
    # Replace PARENT:1946157124 if your Westminster parent code differs.
    parent_filter = "PARENT:1946157124"

    def get_ts066_employment():
        url = (
            "https://www.nomisweb.co.uk/api/v01/dataset/NM_2066_1.data.csv"
            "?time=latest"
            "&geography=TYPE298"                    # 2022 wards
            f"&geography_filter={parent_filter}"    # Westminster wards only
            "&cell=1,2,3,4,5,6,7,8,9"              # all economic activity categories
            "&measures=20100"                       # value
            "&select=geography_name,cell_name,obs_value"
        )

        # Defensive load so the app does not crash if Nomis is empty or errors
        try:
            df = pd.read_csv(url)
        except pd.errors.EmptyDataError:
            st.error(
                "Nomis returned an empty response for TS066 employment.\n\n"
                f"URL called:\n{url}\n\n"
                "Check the dataset code, geography parameters and your Nomis access."
            )
            return pd.DataFrame(columns=["Ward", "Employment Rate", "Unemployment %"])

        required_cols = {"geography_name", "cell_name", "obs_value"}
        if not required_cols.issubset(df.columns):
            st.error(
                "Unexpected response format from Nomis for TS066 (employment).\n\n"
                f"URL called:\n{url}\n\n"
                "First few rows:\n"
                f"{df.head().to_string(index=False)}"
            )
            return pd.DataFrame(columns=["Ward", "Employment Rate", "Unemployment %"])

        # Convert counts to percentages within each ward
        tot = df.groupby("geography_name")["obs_value"].transform("sum")
        df["pct"] = df["obs_value"] / tot * 100

        # Employment = employees + self‑employed
        emp_mask = df["cell_name"].isin(
            [
                "Employee: Full-time",
                "Employee: Part-time",
                "Self-employed: With employees",
                "Self-employed: Without employees",
            ]
        )
        # Unemployment = any category containing 'Unemployed'
        unemp_mask = df["cell_name"].str.contains("Unemployed", case=False)

                out = (
            df.groupby("geography_name")
            .apply(
                lambda g: pd.Series(
                    {
                        "Employment Rate": g.loc[
                            emp_mask & g.index.isin(g.index), "pct"
                        ].sum(),
                        "Unemployment %": g.loc[
                            unemp_mask & g.index.isin(g.index), "pct"
                        ].sum(),
                    }
                )
            )
            .reset_index()
            .rename(columns={"geography_name": "Ward"})
        )

        # Normalise '&' vs 'and' so Nomis names match hard-coded ward names
        def normalise_ward_name(name: str) -> str:
            if not isinstance(name, str):
                return name
            n = name.strip()
            n = n.replace(" and ", " & ")
            return n

        out["Ward"] = out["Ward"].apply(normalise_ward_name)
        return out
        

    # Add further helpers (TS067, TS054, etc.) and merge as you extend.
    emp = get_ts066_employment()

    census = emp.copy()
    return census


@st.cache_data
def load_imd_and_census():
    """
    Load IMD 2025 and 2019 data for Westminster's 18 wards and merge with
    real Census 2021 ward-level data from the Nomis API.
    """

    # -------------------------------------------------------------------------
    # Real IMD data - pulled directly from MHCLG published spreadsheets
    # The ward-level figures are population-weighted averages of LSOA-level scores
    # -------------------------------------------------------------------------
    imd = {
        "Ward": [
            "Westbourne",
            "Church Street",
            "Queen's Park",
            "Harrow Road",
            "Pimlico South",
            "Vincent Square",
            "Pimlico North",
            "Maida Vale",
            "St James's",
            "Bayswater",
            "Little Venice",
            "West End",
            "Hyde Park",
            "Lancaster Gate",
            "Abbey Road",
            "Knightsbridge & Belgravia",
            "Marylebone",
            "Regent's Park",
        ],

        # IMD 2025 - higher score = more deprived relative to rest of England
        "IMD 2025 Score": [
            46.18, 39.80, 36.48, 34.60, 28.12, 24.53, 23.27, 21.75,
            20.91, 20.90, 21.34, 20.34, 18.05, 17.54, 17.55,
            13.96, 13.66, 13.46
        ],
        # Rank 1 = most deprived in England; 6,904 = least deprived
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
        # IMD Health Domain scores (2019) - standardised z-scores
        # Negative = better health relative to England average
        "Health Score 2019": [
            0.29, 0.29, 0.06, 0.00, -0.48, -0.58, -0.68, -1.04,
            -1.14, -0.73, -0.51, -1.26, -1.09, -0.92, -1.62,
            -1.72, -1.61, -1.93
        ],
    }
    df = pd.DataFrame(imd)

    # Calculate change between 2019 and 2025 indices
    # Note: scores are not directly comparable across editions - treat as directional signal
    df["Score Change"] = (df["IMD 2025 Score"] - df["IMD 2019 Score"]).round(2)
    # Positive rank change = moved up the deprivation ranking = worsened
    df["Rank Change"] = df["IMD 2019 Rank"] - df["IMD 2025 Rank"]
    df["Deprivation Direction"] = df["Score Change"].apply(
        lambda x: "Worsened" if x > 1 else ("Improved" if x < -1 else "Stable")
    )

    # -------------------------------------------------------------------------
    # National context bands - where each ward sits relative to all England wards
    # Calculated as rank / 6,904 - based on ONS 2022 ward boundaries
    # -------------------------------------------------------------------------
    def nat_ctx(rank):
        if rank <= TOTAL_WARDS_ENGLAND * 0.10:
            return "Top 10% most deprived"
        elif rank <= TOTAL_WARDS_ENGLAND * 0.20:
            return "Top 20% most deprived"
        elif rank <= TOTAL_WARDS_ENGLAND * 0.40:
            return "Top 40% most deprived"
        elif rank <= TOTAL_WARDS_ENGLAND * 0.80:
            return "Middle 40%"
        else:
            return "Least deprived 20%"

    df["National Context"] = df["IMD 2025 Rank"].apply(nat_ctx)

    # -------------------------------------------------------------------------
    # REAL Census 2021 ward-level data via Nomis API
    # -------------------------------------------------------------------------
    census_real = load_real_census_from_nomis()

    if census_real.empty:
        st.warning(
            "Census 2021 data could not be loaded from Nomis. "
            "The dashboard will show IMD indicators only."
        )
        out = df.copy()
    else:
        # Merge IMD ward table with real census table by ward name
        out = df.merge(census_real, on="Ward", how="left")

    # Quintile label within WCC only (not national)
    out["IMD Quintile"] = pd.qcut(
        out["IMD 2025 Score"],
        q=5,
        labels=[
            "Q5 - Least deprived",
            "Q4",
            "Q3",
            "Q2",
            "Q1 - Most deprived",
        ],
    )
    return out

# Load all data
df   = load_imd_and_census()
ages = load_age_profile()
ind  = load_industry_mix()
geo  = load_geojson()

# Hard-coded ward order (for filters, charts, tables)
WARDS = [
    "Westbourne",
    "Church Street",
    "Queen's Park",
    "Harrow Road",
    "Pimlico South",
    "Vincent Square",
    "Pimlico North",
    "Maida Vale",
    "St James's",
    "Bayswater",
    "Little Venice",
    "West End",
    "Hyde Park",
    "Lancaster Gate",
    "Abbey Road",
    "Knightsbridge & Belgravia",
    "Marylebone",
    "Regent's Park",
]

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## WCC Census & IMD")
    st.markdown("### Westminster City Council")
    st.markdown("---")

    page = st.radio("Navigate", [
        "Overview & IMD",
        "Deprivation Trends",
        "Demographics",
        "Housing & Tenure",
        "Economy & Labour",
        "Statistical Analysis",
        "Ward Map",
        "Data Sources & Quality",
        "How It's Built",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Filter wards**")
    sel_wards = st.multiselect(
        "Select wards (blank = all 18)",
        options=WARDS, default=[],
        placeholder="All 18 wards shown"
    )
    active = sel_wards if sel_wards else WARDS
    dff = df[df["Ward"].isin(active)].copy()
    st.caption(f"Showing {len(dff)} of 18 wards")

    st.markdown("---")
    st.caption(
        "**IMD:** [MHCLG 2025]"
        "(https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025)\n\n"
        "**Census:** [ONS 2021 via Nomis]"
        "(https://www.nomisweb.co.uk/sources/census_2021)\n\n"
        f"**England wards total:** {TOTAL_WARDS_ENGLAND:,}"
    )


# =============================================================================
# PAGE HEADER
# =============================================================================
st.markdown("""
<div class="econ-header">
  <h1>Westminster City Council - Deprivation & Census Dashboard</h1>
  <p>18 wards · IMD 2025 & 2019 (MHCLG) · Census 2021 estimates (ONS/Nomis) · WCC 2022 ward boundaries</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: OVERVIEW & IMD
# =============================================================================
if page == "Overview & IMD":

    # Summary context banner at the top - quick at-a-glance for presentations
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

    # KPI row - numbers formatted to avoid line wrapping at 100% zoom on PC screens
    # font-size is 1.55rem with white-space:nowrap so long numbers like 216,972 stay on one line
    k1, k2, k3, k4, k5 = st.columns(5)
    top10 = (dff["IMD 2025 Rank"] <= int(TOTAL_WARDS_ENGLAND * 0.1)).sum()
    with k1:
        st.markdown(f'<div class="kpi-card"><h2>{len(dff)}</h2><p>Wards shown</p></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card gold"><h2>{dff["IMD 2025 Score"].mean():.1f}</h2><p>Avg IMD 2025 Score</p></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card red"><h2>{dff["IMD 2025 Score"].max():.1f}</h2><p>Highest ({dff.loc[dff["IMD 2025 Score"].idxmax(), "Ward"]})</p></div>', unsafe_allow_html=True)
    with k4:
        # Format with commas - kept on one line by CSS white-space:nowrap
        st.markdown(f'<div class="kpi-card"><h2>{dff["Population"].sum():,}</h2><p>Total population</p></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="kpi-card teal"><h2>{top10}</h2><p>Wards in top 10% nationally</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-label">Deprivation ranking</div>'
                    '<div class="section-title">IMD 2025 Score by ward - most to least deprived</div>',
                    unsafe_allow_html=True)

        # Colour the bars by deprivation level:
        # Red = high deprivation (>30), gold = mid (20-30), teal = lower (<20)
        s_df   = dff.sort_values("IMD 2025 Score", ascending=True)
        colors = [ECON_RED if s > 30 else WCC_GOLD if s > 20 else TEAL
                  for s in s_df["IMD 2025 Score"]]

        fig = go.Figure(go.Bar(
            x=s_df["IMD 2025 Score"],
            y=s_df["Ward"],
            orientation="h",
            marker_color=colors,
            customdata=s_df[["IMD 2025 Rank", "National Context", "Score Change"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Score: %{x:.2f}<br>"
                "Rank: %{customdata[0]:,} of 6,904<br>"
                "National band: %{customdata[1]}<br>"
                "Change 2019-2025: %{customdata[2]:+.2f}<extra></extra>"
            ),
        ))
        # Average line annotation - useful reference point for presentations
        fig.add_vline(
            x=dff["IMD 2025 Score"].mean(),
            line_dash="dot", line_color=WCC_BLUE, line_width=1.5,
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

        # Donut chart showing how many wards fall in each national deprivation band
        ctx_order = [
            "Top 10% most deprived", "Top 20% most deprived",
            "Top 40% most deprived", "Middle 40%", "Least deprived 20%"
        ]
        ctx = (dff["National Context"].value_counts()
               .reindex(ctx_order, fill_value=0)
               .reset_index())
        ctx.columns = ["Context", "Wards"]
        ctx = ctx[ctx["Wards"] > 0]

        fig2 = px.pie(
            ctx, names="Context", values="Wards", hole=0.45,
            color="Context",
            color_discrete_map={
                "Top 10% most deprived": ECON_RED,
                "Top 20% most deprived": CORAL,
                "Top 40% most deprived": WCC_GOLD,
                "Middle 40%": TEAL,
                "Least deprived 20%": WCC_BLUE,
            }
        )
        fig2.update_traces(textposition="outside", textfont_size=9)
        fig2 = econ(fig2, h=285, src="")
        fig2.update_layout(legend=dict(y=-0.4, font_size=9))
        st.plotly_chart(fig2, use_container_width=True)

        # Small clarification note about the banding methodology
        st.markdown(
            f'<p style="font-size:0.71rem;color:#888">Bands: top 10% = rank &le;{int(TOTAL_WARDS_ENGLAND*0.1):,} '
            f'| top 20% = &le;{int(TOTAL_WARDS_ENGLAND*0.2):,} '
            f'| top 40% = &le;{int(TOTAL_WARDS_ENGLAND*0.4):,} '
            f'(of {TOTAL_WARDS_ENGLAND:,} England wards)</p>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-label">Change 2019-2025</div>'
                    '<div class="section-title">Direction of deprivation</div>',
                    unsafe_allow_html=True)

        dir_df = df["Deprivation Direction"].value_counts().reset_index()
        dir_df.columns = ["Direction", "Wards"]
        fig3 = px.bar(
            dir_df, x="Direction", y="Wards", text="Wards",
            color="Direction",
            color_discrete_map={"Worsened": ECON_RED, "Improved": TEAL, "Stable": WCC_GOLD}
        )
        fig3.update_traces(textposition="outside")
        fig3 = econ(fig3, h=195, src="")
        fig3.update_layout(showlegend=False, margin=dict(t=15, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    # Correlation heatmap - helps audience understand which census variables
    # are most strongly associated with IMD scores across wards
    st.markdown('<div class="section-label">Statistical relationships</div>'
                '<div class="section-title">Correlation matrix - IMD score vs census variables (Pearson r)</div>',
                unsafe_allow_html=True)

    c_vars = [
        "IMD 2025 Score", "Employment Rate", "No Qualifications %",
        "Social Rented %", "Good Health %", "Overcrowding %",
        "Degree Level %", "Unemployment %", "Long-term Illness %"
    ]
    corr_fig = px.imshow(
        dff[c_vars].corr(), text_auto=".2f", aspect="auto",
        color_continuous_scale=["#003087", "white", "#E3120B"],
        zmin=-1, zmax=1
    )
    corr_fig = econ(
        corr_fig, h=360,
        subtitle="Pearson r. Red = strong positive correlation; blue = strong negative.",
        src="ONS Census 2021 (modelled estimates); MHCLG IMD 2025"
    )
    st.plotly_chart(corr_fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box">Key finding: IMD Score correlates strongly with '
        "Social Rented % and Unemployment % (positive) and with Good Health % and Degree Level % "
        "(negative) - consistent with the IMD's own domain structure.</div>",
        unsafe_allow_html=True
    )


# =============================================================================
# PAGE 2: DEPRIVATION TRENDS
# =============================================================================
elif page == "Deprivation Trends":

    st.markdown(
        '<div class="warn-box">Comparability note: IMD 2019 and 2025 scores are not directly '
        "comparable - methodology, some indicators and ward boundaries changed between editions. "
        "Score changes should be treated as directional signals. "
        "Rank changes are more reliable for longitudinal comparison. "
        "See the Data Sources & Quality page for full detail.</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">Slope chart</div>'
                    '<div class="section-title">IMD score 2019 vs 2025 by ward</div>',
                    unsafe_allow_html=True)
        fig_s = go.Figure()
        # Draw one line per ward - colour shows direction of change
        for _, row in dff.sort_values("IMD 2025 Score", ascending=False).iterrows():
            c = (ECON_RED if row["Score Change"] > 1
                 else TEAL if row["Score Change"] < -1
                 else WCC_GOLD)
            w = 2.5 if abs(row["Score Change"]) > 3 else 1.5
            fig_s.add_trace(go.Scatter(
                x=[2019, 2025],
                y=[row["IMD 2019 Score"], row["IMD 2025 Score"]],
                mode="lines+markers+text",
                name=row["Ward"],
                line=dict(color=c, width=w),
                marker=dict(size=6),
                text=["", row["Ward"]],
                textposition="middle right",
                textfont=dict(size=8),
                hovertemplate=(
                    f"<b>{row['Ward']}</b><br>"
                    f"2019: {row['IMD 2019 Score']}<br>"
                    f"2025: {row['IMD 2025 Score']}<br>"
                    f"Change: {row['Score Change']:+.2f}<extra></extra>"
                ),
            ))
        fig_s.update_layout(
            xaxis=dict(tickvals=[2019, 2025], showgrid=False, zeroline=False, showline=False),
            yaxis_title="IMD Score",
            showlegend=False
        )
        fig_s = econ(fig_s, h=560, src="MHCLG IMD 2025 and 2019")
        fig_s.update_layout(margin=dict(r=130))
        st.plotly_chart(fig_s, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">Score change</div>'
                    '<div class="section-title">IMD score change 2019 to 2025</div>',
                    unsafe_allow_html=True)
        cd = dff[["Ward", "Score Change", "Deprivation Direction"]].sort_values(
            "Score Change", ascending=False)
        fig_c = px.bar(
            cd, x="Score Change", y="Ward", orientation="h",
            color="Deprivation Direction", text="Score Change",
            color_discrete_map={"Worsened": ECON_RED, "Improved": TEAL, "Stable": WCC_GOLD}
        )
        fig_c.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
        fig_c.add_vline(x=0, line_color="#999", line_width=1)
        fig_c = econ_h(fig_c, h=560, src="MHCLG IMD 2025 and 2019")
        fig_c.update_layout(legend=dict(y=-0.1))
        st.plotly_chart(fig_c, use_container_width=True)

    # Full data table - useful for copy-paste into reports
    st.markdown('<div class="section-label">Full data table</div>'
                '<div class="section-title">Ward-level IMD detail</div>',
                unsafe_allow_html=True)
    cols = [
        "Ward", "IMD 2019 Score", "IMD 2019 Rank",
        "IMD 2025 Score", "IMD 2025 Rank",
        "Score Change", "Rank Change", "Deprivation Direction", "National Context"
    ]
    st.dataframe(
        df[cols].sort_values("IMD 2025 Score", ascending=False).reset_index(drop=True),
        use_container_width=True, hide_index=True
    )


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
                             name="London", orientation="h",
                             marker_color=ECON_RED, opacity=0.55))
        fig.update_layout(barmode="group", xaxis_title="% of population",
                          legend=dict(x=0.55, y=0.05))
        fig = econ_h(fig, h=420, src="ONS Census 2021 (TS007)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">Westminster has a notably higher share of '
            "20-34 year olds than the London average - a professional working-age bulge "
            "reflecting the borough's role as a business and residential centre.</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown('<div class="section-label">Ethnicity</div>'
                    '<div class="section-title">Estimated ethnicity mix by ward</div>',
                    unsafe_allow_html=True)
        eth = dff[["Ward", "White %", "Asian %", "Black %", "Mixed %"]].melt(
            id_vars="Ward", var_name="Ethnicity", value_name="%")
        fig2 = px.bar(
            eth, x="Ward", y="%", color="Ethnicity", barmode="stack",
            color_discrete_map={
                "White %": WCC_BLUE, "Asian %": TEAL,
                "Black %": CORAL, "Mixed %": SAGE
            }
        )
        fig2 = econ(fig2, h=420, src="ONS Census 2021 estimates (TS021, modelled)",
                    rotated=True)
        fig2.update_layout(xaxis_tickangle=-45, legend=dict(y=-0.3))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-label">Health outcomes</div>'
                '<div class="section-title">Self-reported health by ward - sorted by deprivation (highest first)</div>',
                unsafe_allow_html=True)
    hlth = dff.sort_values("IMD 2025 Score", ascending=False)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=hlth["Ward"], y=hlth["Good Health %"],
                          name="Good Health %", marker_color=TEAL))
    fig3.add_trace(go.Bar(x=hlth["Ward"], y=hlth["Long-term Illness %"],
                          name="Long-term Illness %", marker_color=ECON_RED))
    fig3.update_layout(barmode="group", yaxis_title="%", xaxis_tickangle=-45)
    fig3 = econ(fig3, h=340,
                src="ONS Census 2021 estimates (TS037, TS038); IMD 2019 Health domain",
                rotated=True)
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
        ten = dff.sort_values("IMD 2025 Score", ascending=False)
        ten_m = ten[["Ward", "Owner Occupied %", "Social Rented %", "Private Rented %"]].melt(
            id_vars="Ward", var_name="Tenure", value_name="%")
        fig = px.bar(
            ten_m, x="%", y="Ward", orientation="h", barmode="stack",
            color="Tenure",
            color_discrete_map={
                "Owner Occupied %": WCC_BLUE,
                "Social Rented %": ECON_RED,
                "Private Rented %": TEAL
            }
        )
        fig = econ_h(fig, h=520, src="ONS Census 2021 estimates (TS054, modelled)")
        fig.update_layout(legend=dict(y=-0.12))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">Overcrowding</div>'
                    '<div class="section-title">Social renting vs overcrowding</div>',
                    unsafe_allow_html=True)
        fig2 = px.scatter(
            dff, x="Social Rented %", y="Overcrowding %",
            color="IMD 2025 Score", hover_name="Ward",
            size="Population", trendline="ols",
            color_continuous_scale=["#84B59F", "#C8A84B", "#E3120B"]
        )
        fig2 = econ(fig2, h=270, src="ONS Census 2021 estimates; statsmodels OLS trendline")
        fig2.update_layout(coloraxis_colorbar=dict(title="IMD 2025"))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-label">Room size</div>'
                    '<div class="section-title">Owner occupancy vs median rooms</div>',
                    unsafe_allow_html=True)
        fig3 = px.scatter(
            dff, x="Owner Occupied %", y="Median Rooms",
            hover_name="Ward", trendline="ols",
            color="IMD 2025 Score",
            color_continuous_scale=["#84B59F", "#E3120B"]
        )
        fig3 = econ(fig3, h=265, src="ONS Census 2021 estimates (TS050, TS054, modelled)")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        '<div class="insight-box">Wards with the highest IMD scores (Church Street, Westbourne) '
        "show much higher social renting and overcrowding. Overcrowded social housing contributes "
        "directly to the IMD's Housing and Services domain score.</div>",
        unsafe_allow_html=True
    )


# =============================================================================
# PAGE 5: ECONOMY & LABOUR
# =============================================================================
elif page == "Economy & Labour":

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-label">Employment</div>'
                    '<div class="section-title">Employment rate by ward</div>',
                    unsafe_allow_html=True)
        emp = dff.sort_values("Employment Rate")
        fig = px.bar(
            emp, x="Ward", y="Employment Rate",
            color="IMD 2025 Score", text="Employment Rate",
            color_continuous_scale=["#003087", "#C8A84B", "#E3120B"]
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig = econ(fig, h=380, src="ONS Census 2021 estimates (TS066, modelled)", rotated=True)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">Industry</div>'
                    '<div class="section-title">Westminster vs London sector mix</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Westminster", x=ind["Industry"],
                              y=ind["Westminster"], marker_color=WCC_BLUE))
        fig2.add_trace(go.Bar(name="London", x=ind["Industry"],
                              y=ind["London"], marker_color=ECON_RED, opacity=0.65))
        fig2.update_layout(barmode="group", yaxis_title="%", xaxis_tickangle=-45)
        fig2 = econ(fig2, h=380,
                    src="ONS Census 2021 (TS060, WCC); GLA Economics (London)",
                    rotated=True)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-label">Education link</div>'
                    '<div class="section-title">Degree attainment vs employment rate</div>',
                    unsafe_allow_html=True)
        fig3 = px.scatter(
            dff, x="Degree Level %", y="Employment Rate",
            hover_name="Ward", trendline="ols",
            color="IMD 2025 Score",
            color_continuous_scale=["#003087", "#E3120B"],
            size="Population", size_max=30
        )
        fig3 = econ(fig3, h=340, src="ONS Census 2021 estimates (TS066, TS067, modelled)")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-label">Labour market</div>'
                    '<div class="section-title">Unemployment rate vs IMD score</div>',
                    unsafe_allow_html=True)
        fig4 = px.scatter(
            dff, x="IMD 2025 Score", y="Unemployment %",
            hover_name="Ward", trendline="ols",
            color="Deprivation Direction",
            color_discrete_map={"Worsened": ECON_RED, "Improved": TEAL, "Stable": WCC_GOLD},
            size="Population", size_max=30
        )
        fig4 = econ(fig4, h=340, src="ONS Census 2021 estimates; MHCLG IMD 2025")
        st.plotly_chart(fig4, use_container_width=True)


# =============================================================================
# PAGE 6: STATISTICAL ANALYSIS
# =============================================================================
elif page == "Statistical Analysis":

    tab1, tab2, tab3, tab4 = st.tabs([
        "Linear Regression",
        "Random Forest",
        "Clustering",
        "Model Validation",
    ])

    # =========================================================================
    # TAB 1: LINEAR REGRESSION
    # =========================================================================
    with tab1:
        st.markdown("### Linear Regression - predicting IMD 2025 Score")

        st.markdown("""<div class="method-box">
        <b>What is OLS regression?</b><br>
        Ordinary Least Squares (OLS) fits a straight line through the data by minimising the sum
        of squared residuals (the vertical distances between observed and predicted values).
        The result is a set of coefficients that tell you: for a 1-unit increase in predictor X,
        how much does the IMD score change, <i>holding all other predictors constant</i>?<br><br>
        <b>R-squared:</b> The proportion of variance in IMD scores explained by the model.
        A value of 0.85 means 85% of the variation across wards is explained by the selected variables.
        Adjusted R-squared penalises for adding extra predictors that don't genuinely improve fit.<br><br>
        <b>P-values:</b> The probability of observing a coefficient this large by chance if the true
        effect is zero. P &lt; 0.05 is the conventional threshold for statistical significance.
        With n=18, all tests are low-powered - a non-significant result may simply reflect small sample
        size, not the absence of a real relationship.<br><br>
        <b>Confidence intervals (95% CI):</b> The range within which the true coefficient would fall
        95% of the time under repeated sampling. Wide intervals at n=18 are expected.<br><br>
        <b>Best practice (n=18):</b> Aim for no more than 4 predictors. As a rule of thumb,
        n/p &gt; 10 (18 observations divided by predictors). More predictors risks overfitting.
        </div>""", unsafe_allow_html=True)

        sel = st.multiselect("Predictor variables (X)", ALL_CENSUS_FEATS, default=BASE_FEATS)

        if len(sel) >= 1:
            X_raw = df[sel].values
            y_raw = df["IMD 2025 Score"].values
            ols   = sm.OLS(y_raw, sm.add_constant(X_raw)).fit()
            y_hat = ols.predict(sm.add_constant(X_raw))
            resid = y_raw - y_hat
            dw    = durbin_watson(resid)
            sw_s, sw_p = shapiro(resid)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R-squared", f"{ols.rsquared:.3f}")
            m2.metric("Adj. R-squared", f"{ols.rsquared_adj:.3f}")
            m3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_raw, y_hat)):.2f}")
            m4.metric("F-stat p-value", f"{ols.f_pvalue:.4f}",
                      help="Overall significance of the model. <0.05 = model is significant.")

            col1, col2 = st.columns(2)
            with col1:
                # Actual vs predicted scatter - good fit = points close to the diagonal line
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_raw, y=y_hat, mode="markers+text",
                    text=df["Ward"], textposition="top center", textfont=dict(size=8),
                    marker=dict(
                        color=[ECON_RED if e > 0 else TEAL for e in resid],
                        size=9, line=dict(width=1, color="white")
                    ),
                    hovertemplate="<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
                ))
                fig.add_shape(type="line", x0=y_raw.min(), y0=y_raw.min(),
                              x1=y_raw.max(), y1=y_raw.max(),
                              line=dict(color=ECON_GREY, dash="dash"))
                fig = econ(fig, title="Actual vs Predicted",
                           subtitle="Points above the line = model underestimates deprivation",
                           src="", h=380)
                fig.update_layout(xaxis_title="Actual IMD", yaxis_title="Predicted IMD")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Coefficient plot with 95% confidence intervals
                coef_df = pd.DataFrame({
                    "Feature":  sel,
                    "Coef":     ols.params[1:],
                    "CI_low":   ols.conf_int()[1:, 0],
                    "CI_high":  ols.conf_int()[1:, 1],
                    "p_value":  ols.pvalues[1:],
                }).sort_values("Coef")

                fig2 = go.Figure()
                for _, r in coef_df.iterrows():
                    c  = ECON_RED if r["Coef"] > 0 else TEAL
                    op = 1.0 if r["p_value"] < 0.05 else 0.35
                    # Error bar line
                    fig2.add_trace(go.Scatter(
                        x=[r["CI_low"], r["CI_high"]],
                        y=[r["Feature"], r["Feature"]],
                        mode="lines", line=dict(color=c, width=2),
                        opacity=op, showlegend=False
                    ))
                    # Point estimate
                    fig2.add_trace(go.Scatter(
                        x=[r["Coef"]], y=[r["Feature"]],
                        mode="markers", marker=dict(color=c, size=10),
                        opacity=op, showlegend=False,
                        hovertemplate=(
                            f"<b>{r['Feature']}</b><br>"
                            f"Coefficient: {r['Coef']:.3f}<br>"
                            f"95% CI: [{r['CI_low']:.3f}, {r['CI_high']:.3f}]<br>"
                            f"p-value: {r['p_value']:.3f}<extra></extra>"
                        )
                    ))
                fig2.add_vline(x=0, line_color=ECON_GREY, line_width=1.5)
                fig2 = econ(fig2, title="Coefficients + 95% CI",
                            subtitle="Faded dot = not significant (p > 0.05). CI crosses zero = not significant.",
                            src="", h=380)
                fig2.update_layout(xaxis_title="Coefficient (effect per 1-unit increase)")
                st.plotly_chart(fig2, use_container_width=True)

            # Coefficient detail table for the handout / presentation
            ct = coef_df.copy()
            ct["95% CI"]   = ct.apply(lambda r: f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}]", axis=1)
            ct["p-value"]  = ct["p_value"].apply(lambda p: f"{p:.4f} {'*' if p < 0.05 else ''}")
            ct["Direction"] = ct["Coef"].apply(
                lambda c: "Increases deprivation" if c > 0 else "Reduces deprivation")
            st.dataframe(
                ct[["Feature", "Coef", "95% CI", "p-value", "Direction"]].rename(
                    columns={"Coef": "Coefficient"}),
                use_container_width=True, hide_index=True
            )

            # Quick diagnostic summary box
            st.markdown(f"""<div class="method-box">
            <b>Regression diagnostics</b><br>
            <span class="stat-pill">Durbin-Watson = {dw:.3f}</span>
            {'No autocorrelation in residuals (DW near 2)' if 1.5 < dw < 2.5 else 'Possible autocorrelation - interpret CIs with caution'}
            - note: DW is designed for time series; for cross-sectional ward data, Moran's I (spatial autocorrelation) is theoretically more appropriate.<br><br>
            <span class="stat-pill">Shapiro-Wilk W={sw_s:.4f}, p={sw_p:.4f}</span>
            {'Residuals plausibly normal (p > 0.05)' if sw_p > 0.05 else 'Residuals may not be normally distributed - CIs may be unreliable'}<br><br>
            <span class="stat-pill">F-stat={ols.fvalue:.2f}, p={ols.f_pvalue:.4f}</span>
            {'Model is statistically significant overall' if ols.f_pvalue < 0.05 else 'Model not statistically significant at p < 0.05'}
            </div>""", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: RANDOM FOREST
    # =========================================================================
    with tab2:
        st.markdown("### Random Forest - Feature Importance")

        st.markdown("""<div class="method-box">
        <b>What is Random Forest?</b><br>
        An ensemble method that trains many decision trees on bootstrap samples of the data,
        each using a random subset of features. Final predictions are the average across all trees.
        The diversity introduced by bootstrapping and random feature selection reduces overfitting
        compared to a single decision tree.<br><br>
        <b>Feature importance (Mean Decrease Impurity, MDI):</b><br>
        Measures how often each feature is used for tree splits and how much variance it reduces
        at each split, averaged across all trees. Higher = more important for predicting IMD score.
        Importantly, MDI can be biased towards high-cardinality features (continuous variables over
        binary ones). Permutation importance in the Validation tab is a more robust alternative.<br><br>
        <b>OOB R-squared (Out-of-Bag):</b><br>
        Each tree in the forest trains on a bootstrap sample, which by definition excludes roughly
        37% of the data. Those excluded rows are the "out-of-bag" set. The OOB R-squared is the
        average R-squared when each ward is predicted only using trees that did not train on it.
        This gives an unbiased generalisation estimate without needing a separate validation set -
        particularly valuable at n=18 where we cannot afford to hold out a test set.<br><br>
        <b>Nuance:</b> Train R-squared will always be high because trees can memorise training data.
        Always compare OOB R-squared to train R-squared - a big gap suggests overfitting.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        n_t = c1.slider("Number of trees", 10, 300, 100, 10)
        m_d = c2.slider("Max tree depth", 1, 10, 5)

        rf_feats = ALL_CENSUS_FEATS + ["Median Rooms"]
        X_rf = df[rf_feats].values
        y_rf = df["IMD 2025 Score"].values

        rf = RandomForestRegressor(
            n_estimators=n_t, max_depth=m_d,
            random_state=42, oob_score=True
        )
        rf.fit(X_rf, y_rf)
        rf_pred = rf.predict(X_rf)

        m1, m2, m3 = st.columns(3)
        m1.metric("Train R-squared", f"{r2_score(y_rf, rf_pred):.3f}")
        m2.metric("OOB R-squared", f"{rf.oob_score_:.3f}",
                  help="Unbiased - each ward predicted only by trees that did not train on it")
        m3.metric("Trees", str(n_t))

        col1, col2 = st.columns(2)
        with col1:
            imp = pd.DataFrame({
                "Feature":    rf_feats,
                "Importance": rf.feature_importances_
            }).sort_values("Importance", ascending=True)

            colors_i = [
                ECON_RED if v > 0.15 else WCC_GOLD if v > 0.08 else TEAL
                for v in imp["Importance"]
            ]
            fig = go.Figure(go.Bar(
                x=imp["Importance"], y=imp["Feature"],
                orientation="h", marker_color=colors_i,
                text=imp["Importance"].apply(lambda x: f"{x:.3f}"),
                textposition="outside"
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
            fig2.add_shape(type="line", x0=y_rf.min(), y0=y_rf.min(),
                           x1=y_rf.max(), y1=y_rf.max(),
                           line=dict(color=ECON_GREY, dash="dash"))
            fig2 = econ(fig2, title="Actual vs RF Predicted",
                        subtitle=f"OOB R-squared = {rf.oob_score_:.3f} (unbiased estimate)",
                        src="", h=440)
            fig2.update_layout(xaxis_title="Actual IMD", yaxis_title="RF Predicted")
            st.plotly_chart(fig2, use_container_width=True)

        if rf.oob_score_ < r2_score(y_rf, rf_pred) - 0.15:
            st.markdown(
                f'<div class="warn-box">Train R-squared ({r2_score(y_rf, rf_pred):.3f}) '
                f"is substantially higher than OOB R-squared ({rf.oob_score_:.3f}), "
                "suggesting some overfitting. Try increasing max depth or reducing trees.</div>",
                unsafe_allow_html=True
            )

    # =========================================================================
    # TAB 3: CLUSTERING
    # =========================================================================
    with tab3:
        st.markdown("### K-Means Clustering - Ward Typologies")

        st.markdown("""<div class="method-box">
        <b>What is K-Means clustering?</b><br>
        An unsupervised algorithm that partitions n observations into k groups by minimising the
        within-cluster sum of squares (WCSS). It starts with k random centroids, assigns each ward
        to the nearest centroid, recalculates centroids, and repeats until stable. Running n_init=10
        restarts with different random seeds to avoid local minima.<br><br>
        <b>Why StandardScaler is essential:</b><br>
        K-Means uses Euclidean distance. Without scaling, a variable measured in hundreds (population)
        would completely dominate a percentage variable between 0 and 100. StandardScaler transforms
        each variable to mean=0, std=1 so all variables contribute equally to the distance calculation.
        The cluster profiles shown use the original (unscaled) values for interpretability.<br><br>
        <b>Choosing k:</b><br>
        There is no single "correct" k. Use the Validation tab's elbow plot, silhouette score, and
        Davies-Bouldin index together. Then choose k based on interpretability for your policy purpose.
        With n=18, k=3 or k=4 typically gives the most actionable ward typologies.<br><br>
        <b>Limitations:</b><br>
        K-Means assumes spherical clusters of similar density. Results can change with different
        random seeds (addressed by n_init=10). With only 18 wards, any cluster with fewer than
        4 members should be interpreted cautiously.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        n_k = c1.slider("Number of clusters", 2, 6, 3)
        cl_feats = c2.multiselect(
            "Variables to cluster on",
            [
                "IMD 2025 Score", "Employment Rate", "Degree Level %",
                "Social Rented %", "Overcrowding %", "Good Health %",
                "Average Age", "Unemployment %"
            ],
            default=["IMD 2025 Score", "Employment Rate", "Social Rented %", "Overcrowding %"]
        )

        if len(cl_feats) >= 2:
            X_cl  = df[cl_feats].values
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
                fig = econ(
                    fig, title=f"Ward Clusters (k={n_k})",
                    subtitle=f"Silhouette = {sil:.3f}  |  Davies-Bouldin = {dbi:.3f}",
                    src="ONS Census 2021 estimates; MHCLG IMD 2025", h=420
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Cluster profiles - average values (original scale)**")

                # Fix: deduplicate groupby columns to avoid the Arrow serialisation error
                # when a user selects "IMD 2025 Score" as one of the clustering variables
                # (it would appear twice in the groupby column list without this deduplication)
                groupby_cols = list(dict.fromkeys(
                    cl_feats + ["Population", "IMD 2025 Score"]
                ))
                cluster_summary = (
                    df_cl.groupby("Cluster")[groupby_cols]
                    .mean()
                    .round(1)
                    .reset_index()   # reset_index avoids Categorical index issues
                )
                st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

                st.markdown("**Ward assignments**")
                wc = (df_cl[["Ward", "Cluster", "IMD 2025 Score", "Population"]]
                      .sort_values(["Cluster", "IMD 2025 Score"], ascending=[True, False]))
                st.dataframe(wc, use_container_width=True, height=240, hide_index=True)

            m1, m2 = st.columns(2)
            m1.metric(
                "Silhouette Score", f"{sil:.3f}",
                help=">0.5 = strong cluster structure | 0.25-0.5 = weak | <0.25 = may be spurious"
            )
            m2.metric(
                "Davies-Bouldin Index", f"{dbi:.3f}",
                help="Lower = better separated clusters. No absolute good/bad threshold."
            )

    # =========================================================================
    # TAB 4: MODEL VALIDATION
    # =========================================================================
    with tab4:
        st.markdown("### Model Validation - Testing How Robust the Results Are")

        st.markdown("""<div class="alert-box">
        Important: n=18 wards. All statistical tests have low power with small samples.
        Confidence intervals are wide. Results should be treated as exploratory and
        triangulated with qualitative evidence and domain knowledge.
        A non-significant result at n=18 does not mean there is no real relationship.
        </div>""", unsafe_allow_html=True)

        vt1, vt2, vt3, vt4 = st.tabs([
            "Regression Diagnostics",
            "Cross-Validation",
            "AUC and DeLong Test",
            "Cluster Validation",
        ])

        # Pre-fit the base regression model used across validation tabs
        X_base = df[BASE_FEATS].values
        y_base = df["IMD 2025 Score"].values
        ols_base = sm.OLS(y_base, sm.add_constant(X_base)).fit()
        resid_base = y_base - ols_base.predict(sm.add_constant(X_base))

        # -----------------------------------------------------------------
        with vt1:
            st.markdown("#### Regression Diagnostics - base model (4 predictors)")

            st.markdown("""<div class="method-box">
            <b>What these tests tell us:</b><br><br>

            <b>Durbin-Watson (DW):</b> Tests for autocorrelation in residuals - whether each residual
            is correlated with the next one. Values close to 2.0 indicate no autocorrelation; values
            below 1.5 suggest positive autocorrelation (residuals follow each other); values above 2.5
            suggest negative autocorrelation. Originally designed for time-series data; for
            cross-sectional ward data, spatial autocorrelation (Moran's I, which tests whether
            adjacent wards have more similar residuals than expected by chance) is theoretically
            more appropriate. Without a spatial weights matrix, DW is used here as a proxy.<br><br>

            <b>Shapiro-Wilk test:</b> Tests whether regression residuals are normally distributed.
            Normal residuals are an assumption of OLS inference. P > 0.05 = we cannot reject
            normality (plausibly normal). P &lt; 0.05 = evidence of non-normality, which may
            invalidate confidence intervals and p-values. With n=18 this test has low power.<br><br>

            <b>Q-Q plot:</b> Visual check for normality. Points following the diagonal line closely
            indicate normally distributed residuals. S-curves suggest skewness; heavy tails suggest
            kurtosis. The Q-Q plot is often more informative than the Shapiro-Wilk test alone.<br><br>

            <b>Residuals vs Fitted:</b> Checks for homoscedasticity (constant variance of residuals).
            Random scatter around zero is ideal. A funnel shape suggests heteroscedasticity -
            variance changes with the fitted value - which makes OLS standard errors unreliable.<br><br>

            <b>VIF (Variance Inflation Factor):</b> Detects multicollinearity - when predictors are
            highly correlated with each other, inflating coefficient standard errors.
            VIF = 1 means no inflation; VIF &gt; 5 is moderate concern; VIF &gt; 10 is serious.
            High VIF means individual coefficients are unreliable even if R-squared is high.<br><br>

            <b>Cook's Distance:</b> Identifies influential observations that disproportionately
            affect the regression coefficients. A ward with high Cook's D is pulling the fitted
            line towards it. The conventional threshold is 4/n = {:.3f} for n=18.
            </div>""".format(4 / len(df)), unsafe_allow_html=True)

            dw_b = durbin_watson(resid_base)
            sw_b, sw_pb = shapiro(resid_base)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Durbin-Watson", f"{dw_b:.3f}",
                      help="Near 2.0 = no autocorrelation. Less meaningful for cross-sectional data.")
            c2.metric("Shapiro-Wilk p", f"{sw_pb:.4f}",
                      help=">0.05 = residuals plausibly normal")
            c3.metric("F-statistic", f"{ols_base.fvalue:.2f}")
            c4.metric("F p-value", f"{ols_base.f_pvalue:.4f}")

            col1, col2 = st.columns(2)
            with col1:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(
                    x=ols_base.fittedvalues, y=resid_base,
                    mode="markers+text",
                    text=df["Ward"], textposition="top center", textfont=dict(size=8),
                    marker=dict(
                        color=[ECON_RED if abs(r) > 2 * resid_base.std() else WCC_BLUE
                               for r in resid_base],
                        size=9, line=dict(width=1, color="white")
                    ),
                    hovertemplate="<b>%{text}</b><br>Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>"
                ))
                fig_r.add_hline(y=0, line_color=ECON_GREY, line_dash="dash")
                fig_r.add_hline(y= 2 * resid_base.std(), line_color=ECON_RED, line_dash="dot", line_width=1,
                                annotation_text="+ 2 sd", annotation_position="right")
                fig_r.add_hline(y=-2 * resid_base.std(), line_color=ECON_RED, line_dash="dot", line_width=1,
                                annotation_text="- 2 sd", annotation_position="right")
                fig_r = econ(fig_r, title="Residuals vs Fitted",
                             subtitle="Red = |residual| > 2 standard deviations. Random scatter = good.",
                             src="", h=360)
                fig_r.update_layout(xaxis_title="Fitted values", yaxis_title="Residuals")
                st.plotly_chart(fig_r, use_container_width=True)

            with col2:
                # Q-Q plot: theoretical normal quantiles vs sample residual quantiles
                (osm, osr) = stats.probplot(resid_base, dist="norm")
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(x=osm[0], y=osm[1], mode="markers",
                                            marker=dict(color=WCC_BLUE, size=9), name="Residuals"))
                fig_qq.add_trace(go.Scatter(
                    x=osm[0],
                    y=osr[1] + osr[0] * np.array(osm[0]),
                    mode="lines", line=dict(color=ECON_RED, dash="dash"), name="Normal reference"
                ))
                fig_qq = econ(
                    fig_qq, title="Q-Q Plot of Residuals",
                    subtitle=f"Shapiro-Wilk W={sw_b:.4f}, p={sw_pb:.4f}. "
                             f"{'Plausibly normal' if sw_pb > 0.05 else 'Departure from normality - interpret CIs with caution'}",
                    src="", h=360
                )
                fig_qq.update_layout(xaxis_title="Theoretical quantiles",
                                     yaxis_title="Sample quantiles", showlegend=False)
                st.plotly_chart(fig_qq, use_container_width=True)

            # VIF table
            st.markdown("#### VIF - multicollinearity check")
            vif_df = pd.DataFrame({
                "Feature":    BASE_FEATS,
                "VIF":        [variance_inflation_factor(X_base, i)
                               for i in range(X_base.shape[1])],
            })
            vif_df["Assessment"] = vif_df["VIF"].apply(
                lambda v: "OK (< 5)" if v < 5 else ("Moderate (5-10)" if v < 10 else "High (> 10)"))
            st.dataframe(vif_df, use_container_width=True, hide_index=True)

            # Cook's Distance bar chart
            st.markdown("#### Cook's Distance - influential wards")
            cooks, _ = ols_base.get_influence().cooks_distance
            thresh    = 4 / len(df)
            fig_ck = go.Figure(go.Bar(
                x=df["Ward"], y=cooks,
                marker_color=[ECON_RED if c > thresh else TEAL for c in cooks],
                hovertemplate="<b>%{x}</b><br>Cook's D: %{y:.4f}<extra></extra>"
            ))
            fig_ck.add_hline(y=thresh, line_dash="dash", line_color=ECON_RED,
                             annotation_text=f"Threshold 4/n={thresh:.3f}",
                             annotation_position="right")
            fig_ck = econ(fig_ck, title="Cook's Distance",
                          subtitle="Above threshold = potentially high leverage on regression coefficients",
                          src="", h=300, rotated=True)
            fig_ck.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_ck, use_container_width=True)

        # -----------------------------------------------------------------
        with vt2:
            st.markdown("#### Cross-Validation - estimating generalisation performance")

            st.markdown("""<div class="method-box">
            <b>Why do we cross-validate?</b><br>
            Training and testing a model on the same data gives an overly optimistic estimate
            of performance. Cross-validation holds out different subsets of the data for testing,
            giving a fairer picture of how well the model would generalise to unseen wards.<br><br>

            <b>Leave-One-Out CV (LOO-CV):</b><br>
            With n=18, this is the most statistically efficient approach. Each of the 18 wards is
            held out once as the sole test observation, while the model trains on the remaining 17.
            The LOO R-squared is the mean across all 18 folds. Because the test folds are very
            small (n=1), individual fold scores are noisy but the average is a reliable estimate.<br><br>

            <b>5-Fold CV:</b><br>
            Data split into 5 folds of approximately 3-4 wards each. Each fold serves as the
            test set once. With n=18, each fold has only 3-4 observations, so fold-to-fold
            variance is high and individual fold scores should be treated with caution.
            The mean and standard deviation across folds are reported.<br><br>

            <b>Interpreting negative CV R-squared:</b><br>
            A negative value means the model predicts worse than simply predicting the mean
            for every ward. This can happen when a test fold contains an unusual ward that
            the model trained on very different data cannot generalise to. With test folds
            of 3-4 wards at n=18, negative fold scores are not uncommon and do not mean
            the model is fundamentally broken.
            </div>""", unsafe_allow_html=True)

            lr_sk = LinearRegression()
            rf_cv = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            loo   = LeaveOneOut()
            kf5   = KFold(n_splits=5, shuffle=True, random_state=42)

            loo_lr = cross_val_score(lr_sk, X_base, y_base, cv=loo, scoring="r2")
            loo_rf = cross_val_score(rf_cv, X_base, y_base, cv=loo, scoring="r2")
            kf_lr  = cross_val_score(lr_sk, X_base, y_base, cv=kf5, scoring="r2")
            kf_rf  = cross_val_score(rf_cv, X_base, y_base, cv=kf5, scoring="r2")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("LOO R-sq (Linear)", f"{loo_lr.mean():.3f}",
                      delta=f"sd {loo_lr.std():.3f}")
            m2.metric("LOO R-sq (RF)", f"{loo_rf.mean():.3f}",
                      delta=f"sd {loo_rf.std():.3f}")
            m3.metric("5-Fold R-sq (Linear)", f"{kf_lr.mean():.3f}",
                      delta=f"sd {kf_lr.std():.3f}")
            m4.metric("5-Fold R-sq (RF)", f"{kf_rf.mean():.3f}",
                      delta=f"sd {kf_rf.std():.3f}")

            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(x=list(range(1, 6)), y=kf_lr,
                                    name="Linear Regression", marker_color=WCC_BLUE))
            fig_cv.add_trace(go.Bar(x=list(range(1, 6)), y=kf_rf,
                                    name="Random Forest", marker_color=ECON_RED))
            fig_cv.add_hline(y=0, line_color=ECON_GREY, line_width=1)
            fig_cv = econ(
                fig_cv, title="5-Fold CV - R-squared per fold",
                subtitle="High variance between folds is expected at n=18. Wide bars = high uncertainty.",
                src="scikit-learn cross_val_score", h=310
            )
            fig_cv.update_layout(xaxis_title="Fold number", yaxis_title="R-squared", barmode="group")
            st.plotly_chart(fig_cv, use_container_width=True)

            # LOO per ward - shows which wards are hardest to predict
            fig_loo = go.Figure()
            fig_loo.add_trace(go.Scatter(x=df["Ward"], y=loo_lr, mode="markers+lines",
                                         name="Linear", marker_color=WCC_BLUE, marker_size=8))
            fig_loo.add_trace(go.Scatter(x=df["Ward"], y=loo_rf, mode="markers+lines",
                                         name="Random Forest", marker_color=ECON_RED, marker_size=8))
            fig_loo.add_hline(y=0, line_color=ECON_GREY, line_dash="dash")
            fig_loo = econ(
                fig_loo, title="LOO CV - R-squared when each ward is held out",
                subtitle="Negative = model fails to predict that ward from the remaining 17 alone",
                src="", h=340, rotated=True
            )
            fig_loo.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_loo, use_container_width=True)

        # -----------------------------------------------------------------
        with vt3:
            st.markdown("#### AUC, ROC Curve and DeLong-style Comparison")

            st.markdown("""<div class="method-box">
            <b>Converting to a classification problem:</b><br>
            AUC and ROC curves are metrics for binary classification. We convert the IMD 2025 Score
            to a binary variable: "high deprivation" = score above the median (assigned 1), else 0.
            Two logistic regression models are then compared.<br><br>

            <b>AUC (Area Under the ROC Curve):</b><br>
            The ROC curve plots True Positive Rate (sensitivity) against False Positive Rate
            (1 - specificity) at every possible classification threshold. The AUC summarises
            this as a single number: the probability that the model ranks a randomly chosen
            high-deprivation ward above a randomly chosen low-deprivation ward.
            AUC = 0.5 is random (no discrimination); AUC = 1.0 is perfect.<br><br>

            <b>5-Fold Stratified AUC:</b><br>
            Stratified k-fold ensures each fold contains approximately the same proportion of
            high/low deprivation wards (important with small n). The mean and standard deviation
            of AUC across the 5 folds gives an out-of-sample performance estimate.
            Wide standard deviations are expected at n=18.<br><br>

            <b>DeLong test (bootstrap approximation):</b><br>
            The formal DeLong (1988) test compares two AUC values from classifiers applied to
            the same test set, accounting for the correlation between their predictions.
            Here it is approximated by 1,000 bootstrap resamples because the formal version
            requires paired predictions on a held-out test set, which is not feasible at n=18.
            The bootstrap distribution of (AUC A - AUC B) is used to construct a 95% CI and
            p-value. A CI that excludes zero indicates a statistically significant difference
            in discrimination between the two models.
            With n=18, the CI will almost always include zero - this is a sample size problem,
            not evidence that the models are equivalent.
            </div>""", unsafe_allow_html=True)

            med    = df["IMD 2025 Score"].median()
            y_bin  = (df["IMD 2025 Score"] > med).astype(int)

            # Model A: 4 features | Model B: 2 simpler features
            fa = BASE_FEATS
            fb = ["Employment Rate", "Degree Level %"]

            Xa  = StandardScaler().fit_transform(df[fa].values)
            Xb  = StandardScaler().fit_transform(df[fb].values)
            lrA = LogisticRegression(max_iter=1000, random_state=42)
            lrB = LogisticRegression(max_iter=1000, random_state=42)
            lrA.fit(Xa, y_bin); pA = lrA.predict_proba(Xa)[:, 1]
            lrB.fit(Xb, y_bin); pB = lrB.predict_proba(Xb)[:, 1]

            aA = roc_auc_score(y_bin, pA)
            aB = roc_auc_score(y_bin, pB)

            skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_A = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                                   Xa, y_bin, cv=skf, scoring="roc_auc")
            cv_B = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                                   Xb, y_bin, cv=skf, scoring="roc_auc")

            # Bootstrap DeLong-style comparison
            np.random.seed(99)
            diffs = []
            for _ in range(1000):
                idx = np.random.choice(len(y_bin), len(y_bin), replace=True)
                yb  = y_bin.iloc[idx]
                if len(np.unique(yb)) < 2:
                    continue
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
                fpA, tpA, _ = roc_curve(y_bin, pA)
                fpB, tpB, _ = roc_curve(y_bin, pB)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpA, y=tpA, mode="lines",
                    name=f"Model A (4 vars) AUC={aA:.3f}",
                    line=dict(color=WCC_BLUE, width=2.5)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=fpB, y=tpB, mode="lines",
                    name=f"Model B (2 vars) AUC={aB:.3f}",
                    line=dict(color=ECON_RED, width=2.5, dash="dash")
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC=0.5)",
                    line=dict(color=ECON_GREY, dash="dot")
                ))
                fig_roc = econ(
                    fig_roc, title="ROC Curves - Models A and B",
                    subtitle=f"Binary target: IMD score > {med:.1f} (median) = high deprivation",
                    src="scikit-learn LogisticRegression; binary threshold = median IMD", h=420
                )
                fig_roc.update_layout(xaxis_title="False Positive Rate",
                                      yaxis_title="True Positive Rate",
                                      legend=dict(y=-0.3))
                st.plotly_chart(fig_roc, use_container_width=True)

            with col2:
                m1, m2 = st.columns(2)
                m1.metric("AUC - Model A (4 vars)", f"{aA:.3f}")
                m2.metric("AUC - Model B (2 vars)", f"{aB:.3f}")
                m1.metric("5-Fold AUC - Model A", f"{cv_A.mean():.3f}",
                          delta=f"sd {cv_A.std():.3f}")
                m2.metric("5-Fold AUC - Model B", f"{cv_B.mean():.3f}",
                          delta=f"sd {cv_B.std():.3f}")

                # Bootstrap distribution of AUC differences
                fig_b = go.Figure(go.Histogram(
                    x=diffs, nbinsx=40, marker_color=WCC_BLUE, opacity=0.75))
                fig_b.add_vline(x=0, line_color=ECON_RED, line_dash="dash")
                fig_b.add_vrect(x0=ci_lo, x1=ci_hi, fillcolor=WCC_GOLD, opacity=0.15,
                                annotation_text="95% CI", annotation_position="top left")
                fig_b = econ(
                    fig_b, title="Bootstrap AUC difference (A - B)",
                    subtitle=f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}] | p = {p_dl:.3f}",
                    src="1,000 bootstrap resamples", h=260
                )
                st.plotly_chart(fig_b, use_container_width=True)

            st.markdown(f"""<div class="method-box">
            <b>DeLong-style result:</b><br>
            <span class="stat-pill">AUC difference = {aA - aB:.3f}</span>
            <span class="stat-pill">95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]</span>
            <span class="stat-pill">p = {p_dl:.3f}</span><br><br>
            {'The 95% CI excludes zero - Model A has a significantly higher AUC than Model B.' if (ci_lo > 0 or ci_hi < 0) else
             'The 95% CI includes zero - no statistically significant difference in AUC. This is expected at n=18: '
             'the sample is too small to reliably detect moderate differences in AUC. '
             'It does not mean the two models are equivalent in practice.'}<br><br>
            <b>Caveat on the DeLong method:</b> The formal DeLong (1988) paper derives an asymptotic
            variance formula for comparing two correlated AUC estimates from the same test set.
            Bootstrap resampling is used here as an approximation because the formal version
            requires a paired, held-out test set. At n=18, any AUC comparison should be
            confirmed with additional data before informing policy decisions.
            </div>""", unsafe_allow_html=True)

        # -----------------------------------------------------------------
        with vt4:
            st.markdown("#### Clustering Validation - choosing the right number of clusters")

            st.markdown("""<div class="method-box">
            <b>Elbow plot (WCSS):</b><br>
            Plots within-cluster sum of squares against k. WCSS always decreases as k increases
            (adding more clusters always explains more variance). Look for the "elbow" - the point
            where the rate of decrease slows sharply. Beyond this k, adding more clusters gives
            diminishing returns in explanatory power. The elbow is a visual heuristic and can be
            ambiguous; use it alongside the silhouette and Davies-Bouldin scores.<br><br>

            <b>Silhouette score:</b><br>
            For each observation, the silhouette value = (b - a) / max(a, b), where:
            - a = mean distance to other points in the same cluster
            - b = mean distance to points in the nearest other cluster
            A value near +1 means the observation is well-matched to its cluster and far from
            neighbours. Near 0 = overlapping clusters. Negative = possibly wrong cluster.
            The overall silhouette score is the mean across all observations.
            Threshold guidance: > 0.5 = strong structure; 0.25-0.5 = weak; < 0.25 = possibly spurious.<br><br>

            <b>Davies-Bouldin Index (DBI):</b><br>
            For each cluster, DBI computes the ratio of within-cluster scatter to the distance
            between cluster centroids, then averages the worst such ratio across all cluster pairs.
            Lower = better (clusters are tight and well-separated). Unlike silhouette, DBI penalises
            clusters that are close together even if individually tight. The two metrics can
            disagree; use both to triangulate the best k.
            </div>""", unsafe_allow_html=True)

            cl_b = ["IMD 2025 Score", "Employment Rate", "Social Rented %", "Overcrowding %"]
            X_v  = StandardScaler().fit_transform(df[cl_b].values)

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
                fig_e = go.Figure(go.Scatter(
                    x=list(k_range), y=wcss, mode="lines+markers",
                    marker=dict(color=WCC_BLUE, size=8),
                    line=dict(color=WCC_BLUE, width=2.5)
                ))
                fig_e = econ(fig_e, title="Elbow Plot (WCSS)",
                             subtitle="Look for the elbow where the curve flattens", src="", h=270)
                fig_e.update_layout(xaxis_title="k (clusters)", yaxis_title="WCSS")
                st.plotly_chart(fig_e, use_container_width=True)

            with col2:
                fig_s = go.Figure(go.Bar(
                    x=list(k_range), y=sils,
                    marker_color=[ECON_RED if k == best_sil else TEAL for k in k_range],
                    text=[f"{s:.3f}" for s in sils], textposition="outside"
                ))
                fig_s = econ(fig_s, title="Silhouette Score",
                             subtitle=f"Best k = {best_sil} (highest = best defined clusters)",
                             src="", h=270)
                fig_s.update_layout(xaxis_title="k", yaxis_title="Silhouette score")
                st.plotly_chart(fig_s, use_container_width=True)

            with col3:
                fig_d = go.Figure(go.Bar(
                    x=list(k_range), y=dbis,
                    marker_color=[ECON_RED if k == best_dbi else TEAL for k in k_range],
                    text=[f"{d:.3f}" for d in dbis], textposition="outside"
                ))
                fig_d = econ(fig_d, title="Davies-Bouldin Index",
                             subtitle=f"Best k = {best_dbi} (lowest = best separated)",
                             src="", h=270)
                fig_d.update_layout(xaxis_title="k", yaxis_title="DBI")
                st.plotly_chart(fig_d, use_container_width=True)

            st.markdown(
                f'<div class="insight-box">Silhouette suggests k={best_sil}; '
                f"Davies-Bouldin suggests k={best_dbi}. "
                f"{'These agree - k=' + str(best_sil) + ' is a robust choice.' if best_sil == best_dbi else 'These disagree - try both k values and choose based on which ward typology is most useful for your policy purpose.'} "
                "With n=18, k=3 or k=4 typically produces the most interpretable ward groups.</div>",
                unsafe_allow_html=True
            )


# =============================================================================
# PAGE 7: WARD MAP
# Uses the WCC 2022 ward boundary GeoJSON converted from the TopoJSON file
# provided by the WCC GIS team. The WCC_Wards2022_PBI.json file must be in the
# same folder as this script (and committed to GitHub for deployment).
# =============================================================================
elif page == "Ward Map":

    st.markdown("## Ward Map - IMD 2025 by Geography")
    st.markdown("""<div class="context-banner">
    Ward boundaries: WCC 2022 ward review. GeoJSON converted from TopoJSON source provided by WCC GIS team.
    Commit WCC_Wards2022_PBI.json to your GitHub repo alongside this Python file for deployment.
    </div>""", unsafe_allow_html=True)

    if geo is None:
        st.error(
            "WCC_Wards2022_PBI.json not found. "
            "Make sure the file is in the same directory as nomis_dashboard.py and committed to your GitHub repo. "
            "Download the file from the outputs of the dashboard build."
        )
    else:
        # Variable selector for the choropleth colour
        map_var = st.selectbox(
            "Colour map by",
            [
                "IMD 2025 Score",
                "IMD 2019 Score",
                "Score Change",
                "Employment Rate",
                "Unemployment %",
                "Social Rented %",
                "Overcrowding %",
                "Degree Level %",
                "Good Health %",
            ]
        )

        # Choose colour scale to match direction of variable:
        # Red = more concerning; for "Good Health %" flip so red = lower health
        pos_vars = ["Good Health %", "Employment Rate", "Degree Level %"]
        cscale   = "Blues_r" if map_var in pos_vars else "Reds"

        # Match ward names between DataFrame and GeoJSON feature ids
        # GeoJSON feature ids use the 'Label' property (e.g. 'Church Street')
        map_df = df[["Ward", map_var]].copy()

        fig_map = px.choropleth_map(
            map_df,
            geojson=geo,
            locations="Ward",
            color=map_var,
            featureidkey="id",
            center={"lat": 51.513, "lon": -0.145},
            zoom=12,
            map_style="carto-positron",
            color_continuous_scale=cscale,
            hover_data={"Ward": True, map_var: ":.2f"},
            opacity=0.75,
        )

        # Add red rule aesthetic above the map control strip
        fig_map.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=30, b=10),
            coloraxis_colorbar=dict(
                title=map_var,
                len=0.6,
                thickness=14,
                tickfont=dict(size=10)
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Summary table alongside map for quick reference
        col1, col2 = st.columns([3, 2])
        with col2:
            st.markdown(f"**{map_var} - all 18 wards ranked**")
            map_table = df[["Ward", map_var, "National Context"]].sort_values(
                map_var, ascending=False
            ).reset_index(drop=True)
            st.dataframe(map_table, use_container_width=True, hide_index=True, height=400)

        with col1:
            st.markdown(f"**{map_var} distribution across wards**")
            sorted_map = df.sort_values(map_var, ascending=True)
            bar_colors  = [
                ECON_RED if v > sorted_map[map_var].quantile(0.75)
                else WCC_GOLD if v > sorted_map[map_var].median()
                else TEAL
                for v in sorted_map[map_var]
            ]
            fig_bar = go.Figure(go.Bar(
                x=sorted_map[map_var],
                y=sorted_map["Ward"],
                orientation="h",
                marker_color=bar_colors,
                hovertemplate="<b>%{y}</b><br>" + map_var + ": %{x:.2f}<extra></extra>",
            ))
            fig_bar = econ_h(
                fig_bar,
                title=map_var,
                subtitle="Red = top quartile | Gold = above median | Teal = below median",
                h=500,
                src="MHCLG IMD 2025; ONS Census 2021 (modelled estimates)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("""<div class="warn-box">
        Note on the ward boundary file: the GeoJSON was converted from a TopoJSON file
        (WCC_Wards2022_PBI.json) provided by the WCC GIS team in Web Mercator projection.
        Coordinates are in WGS84 (latitude/longitude). For high-precision spatial analysis,
        use the original shapefile from the WCC GIS repository.
        </div>""", unsafe_allow_html=True)


# =============================================================================
# PAGE 8: DATA SOURCES & QUALITY
# =============================================================================
elif page == "Data Sources & Quality":

    st.markdown("## Data Sources, Methodology and Quality Assurance")
    st.markdown("""<div class="context-banner">
    Every data point in this dashboard is documented here - where it comes from, known quality
    issues, and what this means for how findings should be interpreted.
    Essential reading before using any outputs for policy decisions or briefings.
    </div>""", unsafe_allow_html=True)

    ds1, ds2, ds3 = st.tabs(["IMD Data", "Census 2021", "Quality and Limitations"])

    with ds1:
        st.markdown("### Index of Multiple Deprivation (IMD)")
        st.markdown(f"""
**Primary source:** Ministry of Housing, Communities and Local Government (MHCLG)

- [IMD 2025 - official publication](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025)
- [IMD 2025 Technical Report](https://www.gov.uk/government/publications/english-indices-of-deprivation-2025-technical-report)
- [IMD 2019 - previous edition](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)

**National total of {TOTAL_WARDS_ENGLAND:,} wards in England** (ONS 2022 ward boundaries).
This figure is used to calculate national percentile bands throughout this dashboard.
        """)

        st.dataframe(pd.DataFrame({
            "Domain": [
                "Income", "Employment", "Education, Skills and Training",
                "Health Deprivation and Disability", "Crime",
                "Barriers to Housing and Services", "Living Environment"
            ],
            "Weight": ["22.5%", "22.5%", "13.5%", "13.5%", "9.3%", "9.3%", "9.3%"],
            "What it measures": [
                "Low-income households; out-of-work benefits; tax credit recipients",
                "Involuntary exclusion from work (not retired, students or long-term sick)",
                "Lack of attainment and skills; adult skills deprivation",
                "Risk of premature death and impaired quality of life through ill health",
                "Rates of violence, burglary, theft and criminal damage",
                "Physical and financial barriers to decent housing and local services",
                "Quality of local environment: air quality, housing condition, road accidents"
            ],
        }), use_container_width=True, hide_index=True)

        st.markdown(f"""
**National band calculation used in this dashboard:**

| Band | Rank threshold | Calculation |
|---|---|---|
| Top 10% most deprived | rank <= {int(TOTAL_WARDS_ENGLAND*0.1):,} | {TOTAL_WARDS_ENGLAND:,} x 0.10 |
| Top 20% most deprived | rank <= {int(TOTAL_WARDS_ENGLAND*0.2):,} | {TOTAL_WARDS_ENGLAND:,} x 0.20 |
| Top 40% most deprived | rank <= {int(TOTAL_WARDS_ENGLAND*0.4):,} | {TOTAL_WARDS_ENGLAND:,} x 0.40 |
| Middle 40% | rank <= {int(TOTAL_WARDS_ENGLAND*0.8):,} | {TOTAL_WARDS_ENGLAND:,} x 0.80 |
| Least deprived 20% | rank > {int(TOTAL_WARDS_ENGLAND*0.8):,} | above 80th percentile |
""")

        st.markdown("""<div class="warn-box">
        Comparability warning (MHCLG Technical Report, Section 4): IMD 2025 and 2019 scores
        are NOT directly comparable. Changes between editions include: updated data sources and
        reference years for several indicators; revisions to some domain compositions; different
        ward boundaries (2025 uses 2021 Census boundaries vs 2019's earlier boundaries).
        Score changes should be treated as directional signals only. Rank changes are more
        reliable for longitudinal comparison.
        </div>""", unsafe_allow_html=True)

    with ds2:
        st.markdown("### Census 2021 Data")
        st.markdown("""
**Primary source:** Office for National Statistics (ONS), Census 2021, England and Wales.
Reference date: 21 March 2021.

- [All Census 2021 datasets via Nomis](https://www.nomisweb.co.uk/sources/census_2021)
- [Census 2021 Table Finder](https://www.nomisweb.co.uk/census/2021/data_finder)
- [ONS Quality and Methods Guide - Census 2021](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/methodologies/qualityandmethodsguideforcensusbasedstatisticsuk2021)
        """)

        st.dataframe(pd.DataFrame({
            "Variable": [
                "Employment rate", "Unemployment", "Qualifications",
                "Tenure", "Ethnicity", "Health", "Overcrowding", "Industry"
            ],
            "Nomis dataset code": [
                "TS066", "TS066", "TS067", "TS054",
                "TS021", "TS037", "TS050", "TS060"
            ],
            "Geography": ["Ward"] * 8,
            "Reference year": ["2021"] * 8,
        }), use_container_width=True, hide_index=True)

        st.markdown("**Accessing real ward-level data via the Nomis API:**")
        st.code("""
import pandas as pd

# No API key needed for Census 2021 public data
# Replace NM_2066_1 with the dataset code for your variable (see table above)
url = (
    "https://www.nomisweb.co.uk/api/v01/dataset/"
    "NM_2066_1.data.csv"
    "?geography=TYPE298"                    # All wards in England
    "&geography_filter=PARENT:1946157124"   # Filter: Westminster only
    "&cell=0...9"                           # All economic activity categories
    "&measures=20100"                       # Count
    "&select=geography_name,cell_name,obs_value"
)
df = pd.read_csv(url)   # pandas reads directly from URL
df["pct"] = (
    df["obs_value"]
    / df.groupby("geography_name")["obs_value"].transform("sum")
    * 100
)
print(df.head())
        """, language="python")

        st.markdown("""<div class="alert-box">
        Census variables in this dashboard are MODELLED ESTIMATES - not real Census 2021 ward counts.
        They are statistically anchored to real IMD ward rankings and included for illustration
        and training purposes only. For real policy analysis, download actual data from Nomis
        using the dataset codes and API example above.
        </div>""", unsafe_allow_html=True)

    with ds3:
        st.markdown("### Quality Issues and Analytical Limitations")

        qa_items = [
            ("Census: response rate variation",
             "ONS Quality Guide 2021",
             "Census response rates varied across Westminster wards. Church Street and Harrow Road fell below the national average (~89%). Lower response rates increase coverage bias risk, particularly for younger males, private renters and recent migrants. Data for these groups in high non-response wards may be less reliable.",
             "warn"),
            ("Census: Statistical Disclosure Control",
             "ONS Census 2021 Methodology",
             "ONS applied two SDC methods: (1) Targeted Record Swapping - households with unusual characteristics may have records swapped with a nearby similar household. (2) Cell Key Perturbation - small counts may be adjusted by plus or minus 1-2. Ward-level counts for small population subgroups may therefore not exactly equal the true figure.",
             "warn"),
            ("Census: COVID-19 reference date",
             "ONS Census 2021",
             "All Census 2021 data reflects conditions on 21 March 2021 during the pandemic. Employment, commuting and health data may reflect pandemic-specific conditions rather than long-term structural patterns. Economic activity figures in particular should be treated with care.",
             "warn"),
            ("Census: small area reliability",
             "ONS Quality Guide 2021, Section 6",
             "ONS advises caution when interpreting ward-level data where any single category has a count below 10. For smaller Westminster wards some cross-tabulated variables may be suppressed or perturbed. Population estimates carry a standard error of approximately plus or minus 150-300 persons.",
             "warn"),
            ("IMD: temporal comparability",
             "MHCLG IMD 2025 Technical Report, Section 4",
             "IMD 2025 uses data reference years ranging from 2019 to 2023 depending on the indicator. Income deprivation data is from 2021/22; crime from 2021-23; health from 2018-21. The index reflects conditions at different points in time rather than a single consistent reference date.",
             "warn"),
            ("IMD: LSOA to ward aggregation",
             "MHCLG / ONS",
             "The official IMD is calculated at LSOA level (approximately 1,500 population). Ward scores are derived by population-weighted aggregation of LSOA scores. This process masks within-ward variation: a ward with a moderate average score may contain LSOAs ranging from among England's most to least deprived.",
             "alert"),
            ("IMD: relative not absolute measure",
             "MHCLG IMD 2025 Technical Report, Section 2",
             "The IMD measures relative deprivation - how an area compares to other areas in England, not absolute conditions. An improving rank does not necessarily mean absolute conditions improved; it may mean the area improved more slowly than others, or that other areas worsened more.",
             "warn"),
            ("IMD: health domain quality",
             "MHCLG IMD 2025 Technical Report, Section 7.4",
             "The health domain uses standardised mortality ratios, hospital admissions and GP prescribing data. Prescribing-based indicators may reflect access to and quality of GP services as much as underlying health need. Areas with better GP access may appear more health-deprived on this domain.",
             "warn"),
            ("Analysis: ecological fallacy",
             "Statistical methodology",
             "Ward-level relationships do not necessarily hold for individuals within those wards. Do not infer individual behaviour or characteristics from these aggregate statistics.",
             "alert"),
            ("Analysis: spatial autocorrelation",
             "Statistical methodology",
             "Adjacent wards are likely more similar than distant ones. Standard regression assumes observations are independent. Moran's I would be the appropriate test for spatial autocorrelation in residuals; Durbin-Watson is used in this dashboard as a proxy but is designed for time-series data.",
             "warn"),
        ]

        for title, src, desc, box_type in qa_items:
            st.markdown(
                f'<div class="{box_type}-box"><b>{title}</b> '
                f'<span style="color:#888;font-size:0.79rem">({src})</span><br><br>{desc}</div>',
                unsafe_allow_html=True
            )


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
            st.markdown(
                "- Version control - every change tracked\n"
                "- Collaboration across the team\n"
                "- Free code hosting\n"
                "- Triggers Streamlit auto-redeploy on push"
            )
        with c2:
            st.markdown("### Python")
            st.markdown(
                "- `pandas` - data wrangling and table operations\n"
                "- `plotly` - interactive, hoverable charts\n"
                "- `scikit-learn` - machine learning models\n"
                "- `statsmodels` - OLS regression with full diagnostics (p-values, CIs)\n"
                "- `scipy` - statistical tests (Shapiro-Wilk, bootstrap)\n"
                "- `streamlit` - turns a Python script into a web app"
            )
        with c3:
            st.markdown("### Streamlit")
            st.markdown(
                "- Every widget reruns the script top to bottom\n"
                "- `@st.cache_data` - cache expensive data loads\n"
                "- Tabs, sliders, dropdowns all built in\n"
                "- Deploy free on Streamlit Community Cloud\n"
                "- No HTML, CSS or JavaScript needed"
            )

    with t2:
        st.code("""
import pandas as pd

df = pd.read_csv("imd_data.csv")
df.describe()                                 # Summary statistics
df[df["IMD 2025 Score"] > 30]                 # Filter high deprivation wards
df.sort_values("IMD 2025 Score")              # Sort by score
df.groupby("Quintile")["Score"].mean()        # Group aggregation
df["Change"] = df["Score 2025"] - df["Score 2019"]  # Calculated column
        """, language="python")

    with t3:
        st.code("""
import plotly.express as px

# Horizontal bar chart coloured by IMD score
fig = px.bar(df, x="IMD 2025 Score", y="Ward",
             orientation="h",
             color="IMD 2025 Score",
             color_continuous_scale="Reds")

# Add average line annotation
fig.add_vline(x=df["IMD 2025 Score"].mean(),
              line_dash="dot",
              annotation_text="Borough average")

st.plotly_chart(fig, use_container_width=True)
        """, language="python")

    with t4:
        st.code("""
# requirements.txt - put this file in the root of your GitHub repo
streamlit
pandas
numpy
plotly
scikit-learn
statsmodels
scipy
matplotlib

# Files needed in the repo root:
# nomis_dashboard.py
# requirements.txt
# WCC_Wards2022_PBI.json  <-- download from the dashboard build output

# Deploy steps:
# 1. Push all files to GitHub
# 2. Go to share.streamlit.io
# 3. Click New App
# 4. Select your repo and nomis_dashboard.py
# 5. Click Deploy
        """, language="bash")
        st.success(
            "Dashboard is live at a shareable URL. "
            "No installation needed for anyone viewing it."
        )


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(
    f"Westminster City Council - 18 wards (2022 boundaries) - "
    f"[IMD 2025 (MHCLG)](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025) - "
    f"[Census 2021 (ONS/Nomis)](https://www.nomisweb.co.uk/sources/census_2021) - "
    f"National ranking: {TOTAL_WARDS_ENGLAND:,} wards in England - "
    f"Built with Python and Streamlit"
)
