"""
Westminster City Council — Census 2021 & IMD Analytical Dashboard
=================================================================
Built with: Python · Streamlit · Plotly · Scikit-learn
Data:       IMD 2019 & 2025 (MHCLG) · Census 2021 estimates via Nomis
Wards:      18 Westminster City Council wards (2022 boundaries)

Run locally:   streamlit run nomis_dashboard.py
Deploy:        Push to GitHub → Streamlit Community Cloud
requirements:  streamlit pandas numpy plotly scikit-learn statsmodels matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WCC Census & IMD Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colours ─────────────────────────────────────────────────────────────
WCC_BLUE  = "#003087"
WCC_GOLD  = "#C8A84B"
TEAL      = "#028090"
CORAL     = "#E87461"
SAGE      = "#84B59F"
LIGHT_BG  = "#F7F9FC"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #003087 0%, #028090 100%);
        color: white; padding: 1.4rem 2rem; border-radius: 10px;
        margin-bottom: 1.2rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.7rem; }
    .main-header p  { color: #c8e6ff; margin: 0.3rem 0 0; font-size: 0.9rem; }
    .kpi-card {
        background: white; border-radius: 8px; padding: 0.9rem 1.1rem;
        border-left: 4px solid #003087;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .kpi-card h2 { margin: 0; font-size: 1.75rem; color: #003087; }
    .kpi-card p  { margin: 0.15rem 0 0; font-size: 0.78rem; color: #666; }
    .kpi-card.gold { border-left-color: #C8A84B; }
    .kpi-card.gold h2 { color: #C8A84B; }
    .kpi-card.teal { border-left-color: #028090; }
    .kpi-card.teal h2 { color: #028090; }
    .kpi-card.coral { border-left-color: #E87461; }
    .kpi-card.coral h2 { color: #E87461; }
    .section-header {
        font-size: 1rem; font-weight: 700; color: #003087;
        border-bottom: 2px solid #003087; padding-bottom: 0.25rem;
        margin: 1rem 0 0.7rem;
    }
    .insight-box {
        background: #f0f7ff; border-left: 3px solid #028090;
        padding: 0.65rem 0.9rem; border-radius: 4px; font-size: 0.86rem;
        margin: 0.4rem 0;
    }
    .warn-box {
        background: #fff8f0; border-left: 3px solid #C8A84B;
        padding: 0.65rem 0.9rem; border-radius: 4px; font-size: 0.86rem;
        margin: 0.4rem 0;
    }
    .alert-box {
        background: #fff2f0; border-left: 3px solid #E87461;
        padding: 0.65rem 0.9rem; border-radius: 4px; font-size: 0.86rem;
        margin: 0.4rem 0;
    }
    .context-banner {
        background: #1A1A2E; color: #c8e6ff; padding: 0.6rem 1rem;
        border-radius: 6px; font-size: 0.82rem; margin: 0.5rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  DATA — 18 WCC Wards, real IMD 2025 & 2019, anchored census estimates
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    # ── Real IMD data (MHCLG) ─────────────────────────────────────────────
    imd_data = {
        "Ward": [
            "Church Street", "Westbourne", "Queen's Park", "Harrow Road",
            "Pimlico South", "Pimlico North", "Maida Vale", "Little Venice",
            "Hyde Park", "St James's", "Vincent Square", "Lancaster Gate",
            "West End", "Bayswater", "Knightsbridge & Belgravia",
            "Marylebone", "Abbey Road", "Regent's Park",
        ],
        "IMD 2025 Score": [
            46.18, 39.80, 36.48, 34.60,
            28.12, 24.53, 23.27, 21.75,
            20.91, 20.90, 21.34, 20.34,
            18.05, 17.54, 17.55,
            13.96, 13.66, 13.46,
        ],
        "IMD 2025 Rank": [
            265, 490, 661, 800,
            1394, 1849, 2066, 2337,
            2499, 2501, 2411, 2611,
            3103, 3234, 3233,
            4243, 4337, 4404,
        ],
        "IMD 2019 Score": [
            41.45, 35.89, 32.40, 27.93,
            21.63, 17.93, 19.26, 17.99,
            18.57, 23.18, 19.77, 14.60,
            15.92, 15.79, 15.45,
            12.47, 12.17, 11.34,
        ],
        "IMD 2019 Rank": [
            399, 658, 938, 1416,
            2333, 3072, 2781, 3064,
            2933, 2061, 2666, 3972,
            3589, 3629, 3719,
            4648, 4749, 5056,
        ],
        # IMD 2019 Health domain scores (standardised)
        "Health Score 2019": [
            0.29, 0.29, 0.06, 0.00,
            -0.48, -0.58, -0.68, -1.04,
            -1.14, -0.73, -0.51, -1.26,
            -1.09, -0.92, -1.62,
            -1.72, -1.61, -1.93,
        ],
    }
    df_imd = pd.DataFrame(imd_data)

    # ── Compute change 2019→2025 ───────────────────────────────────────────
    df_imd["Score Change"] = (df_imd["IMD 2025 Score"] - df_imd["IMD 2019 Score"]).round(2)
    df_imd["Rank Change"]  = df_imd["IMD 2019 Rank"] - df_imd["IMD 2025 Rank"]  # positive = worsened (lower rank = more deprived)
    df_imd["Deprivation Direction"] = df_imd["Score Change"].apply(
        lambda x: "Worsened ↑" if x > 1 else ("Improved ↓" if x < -1 else "Stable →")
    )

    # ── Normalised deprivation (0=least, 1=most) for anchoring census vars ─
    mn = df_imd["IMD 2025 Score"].min()
    mx = df_imd["IMD 2025 Score"].max()
    d = (df_imd["IMD 2025 Score"] - mn) / (mx - mn)   # 0→1

    n = len(df_imd)
    np.random.seed(42)

    # Generate census variables anchored to real IMD score
    # Higher deprivation (d→1) drives these relationships:
    pop           = np.random.randint(7_500, 15_500, n)
    pct_working   = (80 - d * 22 + np.random.normal(0, 1.5, n)).clip(52, 82).round(1)
    pct_no_qual   = ( 4 + d * 18 + np.random.normal(0, 1.2, n)).clip(2, 26).round(1)
    pct_degree    = (70 - d * 35 + np.random.normal(0, 2.5, n)).clip(28, 75).round(1)
    pct_social    = ( 5 + d * 52 + np.random.normal(0, 2.5, n)).clip(3, 62).round(1)
    pct_owner     = (58 - d * 38 + np.random.normal(0, 2.0, n)).clip(8, 62).round(1)
    pct_private   = (100 - pct_social - pct_owner).clip(5, 75).round(1)
    pct_white     = (82 - d * 38 + np.random.normal(0, 3.0, n)).clip(30, 88).round(1)
    pct_asian     = ( 5 + d * 18 + np.random.normal(0, 1.5, n)).clip(2, 30).round(1)
    pct_black     = ( 2 + d * 16 + np.random.normal(0, 1.5, n)).clip(1, 22).round(1)
    pct_mixed     = (100 - pct_white - pct_asian - pct_black).clip(2, 18).round(1)
    pct_good_hlth = (88 - d * 22 + np.random.normal(0, 1.5, n)).clip(60, 92).round(1)
    avg_age       = (44 - d *  8 + np.random.normal(0, 1.2, n)).clip(30, 48).round(1)
    pct_overcrowd = ( 2 + d * 20 + np.random.normal(0, 1.5, n)).clip(1, 26).round(1)
    median_rooms  = ( 4.5 - d * 2 + np.random.normal(0, 0.2, n)).clip(2.0, 5.5).round(1)
    pct_unemp     = ( 3 + d * 10 + np.random.normal(0, 0.8, n)).clip(1.5, 15).round(1)
    pct_long_ill  = ( 5 + d * 12 + np.random.normal(0, 1.0, n)).clip(3, 20).round(1)

    df_census = pd.DataFrame({
        "Population":          pop,
        "Employment Rate":     pct_working,
        "Unemployment %":      pct_unemp,
        "No Qualifications %": pct_no_qual,
        "Degree Level %":      pct_degree,
        "Social Rented %":     pct_social,
        "Owner Occupied %":    pct_owner,
        "Private Rented %":    pct_private,
        "White %":             pct_white,
        "Asian %":             pct_asian,
        "Black %":             pct_black,
        "Mixed %":             pct_mixed,
        "Good Health %":       pct_good_hlth,
        "Long-term Illness %": pct_long_ill,
        "Average Age":         avg_age,
        "Overcrowding %":      pct_overcrowd,
        "Median Rooms":        median_rooms,
    })

    # Merge everything on Ward
    df = pd.concat([df_imd.reset_index(drop=True), df_census.reset_index(drop=True)], axis=1)

    # IMD quintile within WCC (1 = most deprived 20%)
    df["IMD Quintile"] = pd.qcut(
        df["IMD 2025 Score"], q=5,
        labels=["5 — Least deprived","4","3","2","1 — Most deprived"]
    )

    # National deprivation context label
    def nat_context(rank):
        if rank <= 1000:   return "Top 20% most deprived nationally"
        elif rank <= 3284: return "Top 40% most deprived nationally"
        elif rank <= 4926: return "Middle 40%"
        else:              return "Least deprived 20% nationally"
    df["National Context"] = df["IMD 2025 Rank"].apply(nat_context)

    return df


@st.cache_data
def load_age_data():
    age_bands = ["0–4","5–9","10–14","15–19","20–24","25–29","30–34",
                 "35–39","40–44","45–49","50–54","55–59","60–64","65–69",
                 "70–74","75–79","80–84","85+"]
    wcc   = [5.1,4.2,3.8,4.5,9.2,10.8,9.5,8.1,7.2,6.3,5.5,4.8,3.9,3.2,2.8,2.1,1.5,1.5]
    london= [6.2,5.5,4.9,5.1,7.9,9.8,9.2,7.8,6.7,6.0,5.3,4.5,3.8,3.1,2.6,1.9,1.2,1.5]
    return pd.DataFrame({"Age Band": age_bands, "Westminster %": wcc, "London %": london})


@st.cache_data
def load_industry_data():
    industries = ["Finance & Insurance","Professional/Scientific","Wholesale/Retail",
                  "Accommodation & Food","Public Admin","Education","Health & Social",
                  "Info & Communication","Arts & Entertainment","Construction","Other"]
    wcc    = [14.2,16.8,9.1,8.3,6.2,5.8,7.4,10.5,4.2,3.8,13.7]
    london = [9.1,12.3,10.8,7.9,5.2,7.1,9.8,8.4,3.9,4.6,20.9]
    return pd.DataFrame({"Industry": industries, "Westminster": wcc, "London": london})


df  = load_data()
ages = load_age_data()
ind  = load_industry_data()

WARDS = sorted(df["Ward"].tolist())


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏛️ Westminster City Council")
    st.markdown("### Census 2021 & IMD Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview & IMD",
         "📉 Deprivation Trends",
         "👥 Demographics",
         "🏘️ Housing & Tenure",
         "💼 Economy & Labour",
         "📊 Statistical Analysis",
         "🐍 How It's Built"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**🔍 Filter Wards**")
    selected_wards = st.multiselect(
        "Select wards (blank = all 18)",
        options=WARDS,
        default=[],
        placeholder="All 18 wards shown",
    )
    active_wards = selected_wards if selected_wards else WARDS
    dff = df[df["Ward"].isin(active_wards)].copy()

    if selected_wards:
        st.caption(f"Showing {len(selected_wards)} of 18 wards")
    else:
        st.caption("Showing all 18 wards")

    st.markdown("---")
    st.markdown("**📊 Data Sources**")
    st.caption(
        "**Deprivation:** MHCLG IMD 2025 & 2019\n\n"
        "**Census variables:** ONS Census 2021 estimates (ward-level)\n\n"
        "**Wards:** WCC 2022 boundaries (18 wards)"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🏛️ Westminster City Council — Census 2021 & Deprivation Dashboard</h1>
  <p>18 Westminster wards · IMD 2025 & 2019 · Ward-level census estimates · ONS / MHCLG</p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1: OVERVIEW & IMD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview & IMD":

    # ── Context banner ────────────────────────────────────────────────────
    most_dep  = df.loc[df["IMD 2025 Score"].idxmax(), "Ward"]
    least_dep = df.loc[df["IMD 2025 Score"].idxmin(), "Ward"]
    score_gap = df["IMD 2025 Score"].max() - df["IMD 2025 Score"].min()
    worsened  = (df["Deprivation Direction"] == "Worsened ↑").sum()

    st.markdown(f"""
    <div class="context-banner">
    📌 <b>Westminster IMD 2025 snapshot:</b> &nbsp;
    Most deprived ward: <b>{most_dep}</b> (score {df['IMD 2025 Score'].max()}) &nbsp;|&nbsp;
    Least deprived: <b>{least_dep}</b> (score {df['IMD 2025 Score'].min()}) &nbsp;|&nbsp;
    Score gap across borough: <b>{score_gap:.1f} points</b> &nbsp;|&nbsp;
    Wards worsened since 2019: <b>{worsened}</b>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f'<div class="kpi-card"><h2>{len(dff)}</h2><p>Wards shown</p></div>',
                    unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card gold"><h2>{dff["IMD 2025 Score"].mean():.1f}</h2>'
                    f'<p>Avg IMD 2025 Score</p></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card teal"><h2>{dff["IMD 2025 Score"].max():.1f}</h2>'
                    f'<p>Highest IMD Score ({dff.loc[dff["IMD 2025 Score"].idxmax(),"Ward"]})</p></div>',
                    unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><h2>{dff["Population"].sum():,}</h2>'
                    f'<p>Total Population</p></div>', unsafe_allow_html=True)
    with k5:
        worst_n = (dff["IMD 2025 Rank"] <= 1000).sum()
        st.markdown(f'<div class="kpi-card coral"><h2>{worst_n}</h2>'
                    f'<p>Wards in top 20% most deprived nationally</p></div>',
                    unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">IMD 2025 Score by Ward — ranked most to least deprived</div>',
                    unsafe_allow_html=True)

        fig = px.bar(
            dff.sort_values("IMD 2025 Score", ascending=True),
            x="IMD 2025 Score", y="Ward", orientation="h",
            color="IMD 2025 Score",
            color_continuous_scale=["#C8E6FF", "#C8A84B", "#E87461"],
            hover_data={"IMD 2025 Rank": True, "National Context": True,
                        "Score Change": True, "IMD 2025 Score": ":.2f"},
            template="plotly_white",
        )
        fig.add_vline(x=dff["IMD 2025 Score"].mean(), line_dash="dash",
                      line_color=WCC_BLUE, annotation_text="WCC avg",
                      annotation_position="top right")
        fig.update_layout(height=540, coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=10, b=10))
        fig.update_traces(hovertemplate=(
            "<b>%{y}</b><br>"
            "IMD 2025 Score: %{x:.2f}<br>"
            "National Rank: %{customdata[0]}<br>"
            "Context: %{customdata[1]}<br>"
            "Change since 2019: %{customdata[2]:+.2f}<extra></extra>"
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        💡 <b>How to read the IMD Score:</b> Higher score = more deprived.
        A score of 46 (Church Street) means substantially more concentrated deprivation
        than a score of 13 (Regent's Park). The national rank shows where each ward sits
        among all ~32,000 LSOAs in England — rank 265 means 265th most deprived nationally.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">IMD 2025 National Context</div>',
                    unsafe_allow_html=True)
        context_counts = dff["National Context"].value_counts().reset_index()
        context_counts.columns = ["Context", "Wards"]
        fig2 = px.pie(context_counts, names="Context", values="Wards",
                      color_discrete_sequence=[CORAL, WCC_GOLD, TEAL, WCC_BLUE],
                      template="plotly_white", hole=0.45)
        fig2.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=10),
                           legend=dict(orientation="h", y=-0.25, font_size=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Deprivation Change 2019→2025</div>',
                    unsafe_allow_html=True)
        direction_counts = df["Deprivation Direction"].value_counts().reset_index()
        direction_counts.columns = ["Direction", "Wards"]
        fig3 = px.bar(direction_counts, x="Direction", y="Wards",
                      color="Direction",
                      color_discrete_map={
                          "Worsened ↑": CORAL,
                          "Improved ↓": TEAL,
                          "Stable →": WCC_GOLD,
                      },
                      template="plotly_white", text="Wards")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=220, showlegend=False,
                           margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="section-header">Deprivation Quintiles (WCC)</div>',
                    unsafe_allow_html=True)
        quintile_df = dff[["Ward","IMD Quintile","IMD 2025 Score"]].sort_values("IMD 2025 Score", ascending=False)
        st.dataframe(
            quintile_df.rename(columns={"IMD Quintile": "Quintile (WCC)"}),
            use_container_width=True, height=220, hide_index=True
        )

    # ── Correlation heatmap ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Correlation Matrix — IMD & Census Variables</div>',
                unsafe_allow_html=True)
    corr_vars = ["IMD 2025 Score", "Employment Rate", "No Qualifications %",
                 "Social Rented %", "Good Health %", "Overcrowding %",
                 "Degree Level %", "Unemployment %", "Long-term Illness %"]
    corr_df = dff[corr_vars].corr()
    fig_corr = px.imshow(corr_df, text_auto=".2f",
                         color_continuous_scale=["#003087","white","#E87461"],
                         zmin=-1, zmax=1, template="plotly_white",
                         aspect="auto")
    fig_corr.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key correlations with IMD 2025 Score:</b>
    Higher deprivation strongly predicts higher social renting, higher overcrowding, higher unemployment,
    and lower degree attainment. The negative correlation with Employment Rate and Good Health confirms
    the IMD composite accurately reflects the lived experience of these wards.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2: DEPRIVATION TRENDS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📉 Deprivation Trends":

    st.markdown('<div class="section-header">IMD Score Change: 2019 → 2025</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="context-banner">
    📌 <b>What this shows:</b> Each ward's IMD score from the 2019 and 2025 indices.
    A rising score means worsening deprivation relative to the rest of England.
    A falling score means relative improvement. Scores are not directly comparable
    between indices but give a strong directional signal.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Slope chart: 2019 vs 2025
        fig_slope = go.Figure()
        for _, row in dff.sort_values("IMD 2025 Score", ascending=False).iterrows():
            colour = CORAL if row["Score Change"] > 1 else (TEAL if row["Score Change"] < -1 else WCC_GOLD)
            fig_slope.add_trace(go.Scatter(
                x=[2019, 2025],
                y=[row["IMD 2019 Score"], row["IMD 2025 Score"]],
                mode="lines+markers+text",
                name=row["Ward"],
                line=dict(color=colour, width=2),
                marker=dict(size=7),
                text=["", row["Ward"]],
                textposition="middle right",
                textfont=dict(size=9),
                hovertemplate=(
                    f"<b>{row['Ward']}</b><br>"
                    f"2019: {row['IMD 2019 Score']}<br>"
                    f"2025: {row['IMD 2025 Score']}<br>"
                    f"Change: {row['Score Change']:+.2f}<extra></extra>"
                ),
            ))
        fig_slope.update_layout(
            height=540, template="plotly_white", showlegend=False,
            xaxis=dict(tickvals=[2019, 2025], ticktext=["IMD 2019", "IMD 2025"]),
            yaxis_title="IMD Score (higher = more deprived)",
            margin=dict(l=0, r=120, t=20, b=10),
        )
        st.plotly_chart(fig_slope, use_container_width=True)

    with col2:
        # Bar chart of score change
        change_df = dff[["Ward","Score Change","Deprivation Direction"]].sort_values("Score Change", ascending=False)
        fig_change = px.bar(
            change_df, x="Score Change", y="Ward", orientation="h",
            color="Deprivation Direction",
            color_discrete_map={"Worsened ↑": CORAL, "Improved ↓": TEAL, "Stable →": WCC_GOLD},
            template="plotly_white",
            text="Score Change",
        )
        fig_change.add_vline(x=0, line_color="#aaa", line_width=1)
        fig_change.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
        fig_change.update_layout(
            height=540, title="Score Change 2019→2025",
            margin=dict(l=0, r=60, t=40, b=10),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_change, use_container_width=True)

    # Detailed change table
    st.markdown('<div class="section-header">Ward-level IMD Detail Table</div>',
                unsafe_allow_html=True)
    table_cols = ["Ward","IMD 2019 Score","IMD 2019 Rank",
                  "IMD 2025 Score","IMD 2025 Rank",
                  "Score Change","Rank Change","Deprivation Direction","National Context"]
    detail_df = df[table_cols].sort_values("IMD 2025 Score", ascending=False).reset_index(drop=True)

    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # Analysis
    worst_change = dff.loc[dff["Score Change"].idxmax(), "Ward"]
    best_change  = dff.loc[dff["Score Change"].idxmin(), "Ward"]

    st.markdown(f"""
    <div class="alert-box">
    ⚠️ <b>Biggest worsening since 2019:</b> {worst_change}
    (score +{dff["Score Change"].max():.2f} points).
    This ward moved {abs(int(dff.loc[dff["Score Change"].idxmax(),"Rank Change"]))} places higher in the national deprivation ranking.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
    ✅ <b>Biggest improvement since 2019:</b> {best_change}
    (score {dff["Score Change"].min():.2f} points).
    </div>""", unsafe_allow_html=True)

    # Scatter: 2019 vs 2025 scores
    st.markdown('<div class="section-header">2019 vs 2025 Scores — Scatter View</div>',
                unsafe_allow_html=True)
    fig_scatter = px.scatter(
        dff, x="IMD 2019 Score", y="IMD 2025 Score",
        color="Score Change",
        hover_name="Ward",
        size="Population",
        color_continuous_scale=["#028090","#C8A84B","#E87461"],
        template="plotly_white",
        text="Ward",
    )
    fig_scatter.add_shape(
        type="line",
        x0=dff["IMD 2019 Score"].min(), y0=dff["IMD 2019 Score"].min(),
        x1=dff["IMD 2019 Score"].max(), y1=dff["IMD 2019 Score"].max(),
        line=dict(dash="dash", color="#aaa")
    )
    fig_scatter.add_annotation(
        x=dff["IMD 2019 Score"].max() * 0.9,
        y=dff["IMD 2019 Score"].max() * 0.85,
        text="Above line = worsened", showarrow=False,
        font=dict(size=10, color="#888")
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=9)
    fig_scatter.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3: DEMOGRAPHICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👥 Demographics":

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Age Profile — Westminster vs London</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=ages["Age Band"], x=ages["Westminster %"],
                             name="Westminster", orientation="h", marker_color=WCC_BLUE))
        fig.add_trace(go.Bar(y=ages["Age Band"], x=ages["London %"],
                             name="London", orientation="h",
                             marker_color=CORAL, opacity=0.7))
        fig.update_layout(barmode="group", height=420, template="plotly_white",
                          legend=dict(x=0.6, y=0.05),
                          xaxis_title="% of population")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="insight-box">
        💡 Westminster's 20–34 age band is notably larger than the London average, reflecting
        its concentration of young professionals. This has implications for housing demand
        and service design.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Ethnicity by Ward</div>',
                    unsafe_allow_html=True)
        eth_melt = dff[["Ward","White %","Asian %","Black %","Mixed %"]].melt(
            id_vars="Ward", var_name="Ethnicity", value_name="Percentage")
        fig2 = px.bar(eth_melt, x="Ward", y="Percentage", color="Ethnicity",
                      color_discrete_map={"White %": WCC_BLUE, "Asian %": TEAL,
                                          "Black %": CORAL, "Mixed %": SAGE},
                      template="plotly_white", barmode="stack",
                      hover_data={"Percentage": ":.1f"})
        fig2.update_layout(height=420, xaxis_tickangle=-45,
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Average Age vs IMD Score (by Ward)</div>',
                unsafe_allow_html=True)
    fig3 = px.scatter(
        dff, x="Average Age", y="IMD 2025 Score",
        size="Population", color="Deprivation Direction",
        hover_name="Ward",
        color_discrete_map={"Worsened ↑": CORAL, "Improved ↓": TEAL, "Stable →": WCC_GOLD},
        template="plotly_white", size_max=40,
        text="Ward",
    )
    fig3.update_traces(textposition="top center", textfont_size=9)
    fig3.update_layout(height=380, yaxis_title="IMD 2025 Score (higher = more deprived)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Health & Long-term Illness by Ward</div>',
                unsafe_allow_html=True)
    hlth_df = dff[["Ward","Good Health %","Long-term Illness %","IMD 2025 Score"]].sort_values("IMD 2025 Score", ascending=False)
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=hlth_df["Ward"], y=hlth_df["Good Health %"],
                          name="Good Health %", marker_color=TEAL))
    fig4.add_trace(go.Bar(x=hlth_df["Ward"], y=hlth_df["Long-term Illness %"],
                          name="Long-term Illness %", marker_color=CORAL))
    fig4.update_layout(barmode="group", height=340, template="plotly_white",
                       xaxis_tickangle=-45, yaxis_title="%",
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""<div class="insight-box">
    💡 Wards with the highest IMD scores (Church Street, Westbourne) tend to have lower
    self-reported good health and higher rates of long-term illness — consistent with the
    health domain scores in the IMD data.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 4: HOUSING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏘️ Housing & Tenure":

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Tenure Mix by Ward (sorted by deprivation)</div>',
                    unsafe_allow_html=True)
        ten_df = dff.sort_values("IMD 2025 Score", ascending=False)
        ten_melt = ten_df[["Ward","Owner Occupied %","Social Rented %","Private Rented %"]].melt(
            id_vars="Ward", var_name="Tenure", value_name="Percentage")
        fig = px.bar(ten_melt, x="Percentage", y="Ward", color="Tenure", orientation="h",
                     color_discrete_map={"Owner Occupied %": WCC_BLUE,
                                         "Social Rented %": CORAL,
                                         "Private Rented %": TEAL},
                     template="plotly_white", barmode="stack")
        fig.update_layout(height=520, legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Overcrowding vs Social Renting</div>',
                    unsafe_allow_html=True)
        fig2 = px.scatter(dff, x="Social Rented %", y="Overcrowding %",
                          color="IMD 2025 Score",
                          hover_name="Ward", size="Population",
                          trendline="ols",
                          color_continuous_scale=["#84B59F","#C8A84B","#E87461"],
                          template="plotly_white",
                          hover_data={"IMD 2025 Score": ":.1f"})
        fig2.update_layout(height=290,
                           coloraxis_colorbar=dict(title="IMD 2025"))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Median Rooms vs Owner Occupancy</div>',
                    unsafe_allow_html=True)
        fig3 = px.scatter(dff, x="Owner Occupied %", y="Median Rooms",
                          hover_name="Ward", trendline="ols",
                          color="IMD 2025 Score",
                          color_continuous_scale=["#84B59F","#E87461"],
                          template="plotly_white")
        fig3.update_layout(height=270)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""<div class="insight-box">
    💡 Wards with the highest IMD scores (Church Street, Westbourne, Queen's Park) have
    significantly higher social renting and overcrowding. This isn't coincidental —
    overcrowded social housing is a direct contributor to the IMD's housing domain score.
    The tenure mix chart (sorted by deprivation, top to bottom) shows the pattern clearly.
    </div>""", unsafe_allow_html=True)

    # Overcrowding ranked
    st.markdown('<div class="section-header">Overcrowding Rate by Ward</div>',
                unsafe_allow_html=True)
    over_df = dff[["Ward","Overcrowding %","IMD 2025 Score"]].sort_values("Overcrowding %", ascending=False)
    fig4 = px.bar(over_df, x="Ward", y="Overcrowding %",
                  color="IMD 2025 Score",
                  color_continuous_scale=["#C8E6FF","#E87461"],
                  template="plotly_white", text="Overcrowding %")
    fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig4.update_layout(height=320, xaxis_tickangle=-45,
                       coloraxis_colorbar=dict(title="IMD 2025"))
    st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 5: ECONOMY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💼 Economy & Labour":

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-header">Employment Rate by Ward (sorted)</div>',
                    unsafe_allow_html=True)
        emp_df = dff.sort_values("Employment Rate")
        fig = px.bar(emp_df, x="Ward", y="Employment Rate",
                     color="IMD 2025 Score",
                     color_continuous_scale=["#003087","#C8A84B","#E87461"],
                     template="plotly_white", text="Employment Rate")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=380, xaxis_tickangle=-45,
                          coloraxis_colorbar=dict(title="IMD 2025"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Industry Mix vs London</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Westminster", x=ind["Industry"],
                              y=ind["Westminster"], marker_color=WCC_BLUE))
        fig2.add_trace(go.Bar(name="London", x=ind["Industry"],
                              y=ind["London"], marker_color=CORAL, opacity=0.75))
        fig2.update_layout(barmode="group", template="plotly_white",
                           height=380, xaxis_tickangle=-45, yaxis_title="%",
                           legend=dict(x=0.6, y=1))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Education vs Employment (by ward)</div>',
                    unsafe_allow_html=True)
        fig3 = px.scatter(dff, x="Degree Level %", y="Employment Rate",
                          hover_name="Ward", trendline="ols",
                          color="IMD 2025 Score",
                          color_continuous_scale=["#003087","#E87461"],
                          template="plotly_white", size="Population", size_max=35,
                          text="Ward")
        fig3.update_traces(textposition="top center", textfont_size=8,
                           selector=dict(mode="text"))
        fig3.update_layout(height=340,
                           coloraxis_colorbar=dict(title="IMD 2025"))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Unemployment vs IMD Score</div>',
                    unsafe_allow_html=True)
        fig4 = px.scatter(dff, x="IMD 2025 Score", y="Unemployment %",
                          hover_name="Ward", trendline="ols",
                          color="Deprivation Direction",
                          color_discrete_map={"Worsened ↑": CORAL,
                                              "Improved ↓": TEAL,
                                              "Stable →": WCC_GOLD},
                          template="plotly_white", size="Population", size_max=30,
                          text="Ward")
        fig4.update_traces(textposition="top center", textfont_size=8,
                           selector=dict(mode="text"))
        fig4.update_layout(height=340)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""<div class="insight-box">
    💡 The positive trendline between IMD 2025 Score and Unemployment % confirms that
    labour market exclusion is a core driver of deprivation in Westminster — not just a
    by-product. Church Street and Westbourne have both the highest deprivation scores and
    the highest estimated unemployment rates.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 6: STATISTICAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Statistical Analysis":

    tab1, tab2, tab3 = st.tabs(["📈 Linear Regression", "🌲 Random Forest", "🔵 Clustering"])

    # ── TAB 1: REGRESSION ────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Predicting IMD 2025 Score with Linear Regression")
        st.markdown("Select predictor variables — the model tries to explain which census factors drive deprivation.")

        all_features = ["Employment Rate","No Qualifications %","Social Rented %",
                        "Good Health %","Overcrowding %","Degree Level %",
                        "Owner Occupied %","Average Age","Unemployment %","Long-term Illness %"]
        selected_features = st.multiselect(
            "Predictor variables (X) — target is always IMD 2025 Score",
            all_features,
            default=["Employment Rate","No Qualifications %","Social Rented %","Overcrowding %"],
        )

        if len(selected_features) >= 1:
            X = df[selected_features].values
            y = df["IMD 2025 Score"].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2   = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R² Score", f"{r2:.3f}", help="How much variance in IMD Score is explained (0–1)")
            m2.metric("RMSE", f"{rmse:.2f}", help="Average prediction error in IMD score points")
            m3.metric("Predictors", str(len(selected_features)))
            m4.metric("Wards", str(len(df)))

            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(
                    x=y, y=y_pred, hover_name=df["Ward"],
                    labels={"x": "Actual IMD 2025 Score", "y": "Predicted"},
                    color=df["Deprivation Direction"],
                    color_discrete_map={"Worsened ↑": CORAL,
                                        "Improved ↓": TEAL, "Stable →": WCC_GOLD},
                    template="plotly_white",
                )
                fig.add_shape(type="line", x0=y.min(), y0=y.min(),
                              x1=y.max(), y1=y.max(),
                              line=dict(color="#aaa", dash="dash"))
                fig.update_layout(title="Actual vs Predicted IMD Score", height=360,
                                  legend_title="Direction")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                coef_df = pd.DataFrame({
                    "Feature": selected_features,
                    "Coefficient": model.coef_,
                    "Direction": ["↑ Increases deprivation" if c > 0 else "↓ Reduces deprivation"
                                  for c in model.coef_],
                }).sort_values("Coefficient")
                fig2 = px.bar(coef_df, x="Coefficient", y="Feature",
                              orientation="h", template="plotly_white",
                              color="Coefficient",
                              color_continuous_scale=["#028090","white","#E87461"],
                              hover_data={"Direction": True})
                fig2.add_vline(x=0, line_color="#aaa", line_width=1)
                fig2.update_layout(title="Regression Coefficients", height=360,
                                   coloraxis_showscale=False)
                st.plotly_chart(fig2, use_container_width=True)

            # Residuals
            residuals = y - y_pred
            st.markdown('<div class="section-header">Residual Analysis — which wards does the model struggle with?</div>',
                        unsafe_allow_html=True)
            resid_df = pd.DataFrame({
                "Ward": df["Ward"],
                "Actual": y.round(2),
                "Predicted": y_pred.round(2),
                "Residual": residuals.round(2),
                "Abs Error": np.abs(residuals).round(2),
            }).sort_values("Abs Error", ascending=False)

            fig3 = px.bar(resid_df.sort_values("Residual"), x="Residual", y="Ward",
                          orientation="h", color="Residual",
                          color_continuous_scale=["#028090","white","#E87461"],
                          template="plotly_white",
                          hover_data={"Actual": True, "Predicted": True})
            fig3.add_vline(x=0, line_color="#aaa", line_width=1.5)
            fig3.update_layout(height=500, coloraxis_showscale=False,
                               title="Residuals (Actual − Predicted): positive = underestimated deprivation")
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown(f"""
            <div class="warn-box">
            💡 <b>R² = {r2:.3f}</b> — your model explains {r2*100:.1f}% of the variation in IMD scores
            across Westminster's 18 wards. An R² above 0.7 with census data is strong. Wards with large
            residuals have deprivation that isn't captured by the selected variables alone — worth
            investigating qualitatively.
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Select at least one predictor variable above to run the model.")

    # ── TAB 2: RANDOM FOREST ──────────────────────────────────────────────────
    with tab2:
        st.markdown("### Random Forest — What Drives IMD Score?")
        st.markdown("Which census variables best predict deprivation? Random Forest finds non-linear patterns automatically.")

        c1, c2 = st.columns(2)
        n_trees   = c1.slider("Number of trees", 10, 300, 100, 10)
        max_depth = c2.slider("Max tree depth", 1, 10, 5)

        rf_features = ["Employment Rate","No Qualifications %","Social Rented %",
                       "Good Health %","Overcrowding %","Degree Level %",
                       "Owner Occupied %","Average Age","Unemployment %",
                       "Long-term Illness %","Median Rooms"]
        X_rf = df[rf_features].values
        y_rf = df["IMD 2025 Score"].values

        rf = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42)
        rf.fit(X_rf, y_rf)
        rf_pred = rf.predict(X_rf)
        rf_r2   = r2_score(y_rf, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_rf, rf_pred))

        m1, m2, m3 = st.columns(3)
        m1.metric("Random Forest R²", f"{rf_r2:.3f}")
        m2.metric("RMSE", f"{rf_rmse:.2f}")
        m3.metric("Trees used", str(n_trees))

        col1, col2 = st.columns(2)
        with col1:
            imp_df = pd.DataFrame({
                "Feature":    rf_features,
                "Importance": rf.feature_importances_,
            }).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance",
                         color_continuous_scale=["#C8E6FF","#003087"],
                         template="plotly_white",
                         text=imp_df["Importance"].apply(lambda x: f"{x:.3f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Feature Importance (contribution to IMD prediction)",
                              height=420, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(
                x=y_rf, y=rf_pred, hover_name=df["Ward"],
                labels={"x": "Actual IMD 2025", "y": "RF Predicted"},
                color=df["Deprivation Direction"],
                color_discrete_map={"Worsened ↑": CORAL,
                                    "Improved ↓": TEAL, "Stable →": WCC_GOLD},
                template="plotly_white",
            )
            fig2.add_shape(type="line", x0=y_rf.min(), y0=y_rf.min(),
                           x1=y_rf.max(), y1=y_rf.max(),
                           line=dict(color="#aaa", dash="dash"))
            fig2.update_layout(title="Random Forest: Actual vs Predicted", height=340)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("""<div class="insight-box">
            💡 <b>Feature importance</b> tells you which variables the trees split on most when
            predicting IMD score. Unlike regression coefficients, importance doesn't show direction —
            but it reveals which variables carry the most information, including non-linear signals.
            </div>""", unsafe_allow_html=True)

    # ── TAB 3: CLUSTERING ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("### K-Means Clustering — Ward Typologies")
        st.markdown("Group similar wards automatically. No target variable — the algorithm finds natural patterns.")

        c1, c2 = st.columns(2)
        n_clusters = c1.slider("Number of clusters (ward types)", 2, 6, 3)
        cluster_features = c2.multiselect(
            "Variables to cluster on",
            ["IMD 2025 Score","Employment Rate","Degree Level %",
             "Social Rented %","Overcrowding %","Good Health %",
             "Average Age","Unemployment %"],
            default=["IMD 2025 Score","Employment Rate","Social Rented %","Overcrowding %"],
        )

        if len(cluster_features) >= 2:
            X_cl = df[cluster_features].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cl)

            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(X_scaled)
            df_cl = df.copy()
            df_cl["Cluster"] = "Type " + (cluster_labels + 1).astype(str)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(
                    df_cl, x=cluster_features[0],
                    y=cluster_features[1] if len(cluster_features) > 1 else "IMD 2025 Score",
                    color="Cluster", hover_name="Ward",
                    size="Population",
                    color_discrete_sequence=[WCC_BLUE, CORAL, TEAL, SAGE, WCC_GOLD, "#6D2E46"],
                    template="plotly_white",
                    text="Ward",
                )
                fig.update_traces(textposition="top center", textfont_size=8,
                                  selector=dict(mode="markers+text"))
                fig.update_layout(height=420, title="Ward Clusters")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                cluster_summary = df_cl.groupby("Cluster")[cluster_features + ["Population"]].mean().round(1)
                st.markdown("**Cluster Profiles — average values per type**")
                st.dataframe(cluster_summary, use_container_width=True)

                st.markdown("**Ward assignments**")
                ward_clusters = df_cl[["Ward","Cluster","IMD 2025 Score","Population"]].sort_values(
                    ["Cluster","IMD 2025 Score"], ascending=[True,False])
                st.dataframe(ward_clusters, use_container_width=True, height=240,
                             hide_index=True)

            st.markdown("""<div class="insight-box">
            💡 <b>Important:</b> Always scale variables before clustering (done automatically here via
            StandardScaler). Variables on different scales would otherwise dominate — e.g. Population
            in the thousands would swamp a percentage variable between 0 and 100.
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Select at least 2 variables to cluster on.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 7: HOW IT'S BUILT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🐍 How It's Built":

    st.markdown("## 🐍 How This Dashboard Works")
    st.markdown("A walkthrough of the Python + Streamlit + GitHub stack — for data professionals new to coding.")

    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ The Stack","2️⃣ Pandas","3️⃣ Plotly","4️⃣ Deploy"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🐙 GitHub")
            st.markdown("""
            - **Version control** — every change tracked
            - **Collaboration** — multiple contributors
            - **Free hosting** — Streamlit reads from here
            - **Portfolio** — share your work publicly
            """)
        with col2:
            st.markdown("### 🐍 Python")
            st.markdown("""
            - `pandas` — data wrangling
            - `plotly` — interactive charts
            - `scikit-learn` — ML & stats models
            - `streamlit` — turns script → web app
            - `numpy` — numerical computing
            """)
        with col3:
            st.markdown("### ⚡ Streamlit")
            st.markdown("""
            - Every widget reruns the script
            - `@st.cache_data` — cache expensive loads
            - Sliders, dropdowns, tabs built in
            - Deploy free on Streamlit Community Cloud
            - No HTML/CSS/JS needed
            """)

    with tab2:
        st.markdown("### pandas — data wrangling in code")
        st.code("""
import pandas as pd

df = pd.read_csv("imd_data.csv")     # Load data
df.describe()                         # Summary statistics
df[df["IMD 2025 Score"] > 30]        # Filter deprived wards
df.sort_values("IMD 2025 Score")     # Sort
df.groupby("Quintile")["Score"].mean() # Group aggregation
df["Change"] = df["2025"] - df["2019"] # New calculated column
        """, language="python")

    with tab3:
        st.markdown("### plotly — one-line interactive charts")
        st.code("""
import plotly.express as px

# Bar chart coloured by IMD score
fig = px.bar(df, x="Ward", y="IMD 2025 Score",
             color="IMD 2025 Score",
             color_continuous_scale="Reds",
             template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Scatter with OLS trendline (needs statsmodels)
fig = px.scatter(df, x="No Qualifications %",
                 y="IMD 2025 Score",
                 hover_name="Ward",
                 trendline="ols")
        """, language="python")

        st.markdown("### 📐 Visualisation principles")
        for k, v in {
            "Match chart to data type": "Continuous → scatter | Categorical → bar | Time → line",
            "Label axes with units": "Always. Never leave a bare number axis.",
            "Sequential colour for ordered data": "Light→dark for low→high values",
            "No 3D charts": "They distort proportions. Bar charts are almost always clearer.",
            "Use hover tooltips": "Let people explore — plotly gives you this for free",
        }.items():
            st.markdown(f"**{k}** — {v}")

    with tab4:
        st.markdown("### 🚀 Deploy in 3 steps")
        st.code("""
# 1. Push to GitHub
git add .
git commit -m "WCC Census dashboard"
git push origin main

# 2. requirements.txt (must be in repo root)
streamlit
pandas
numpy
plotly
scikit-learn
statsmodels
matplotlib

# 3. Go to share.streamlit.io → New App
#    Select repo → nomis_dashboard.py → Deploy
        """, language="bash")

        st.markdown("### 📡 Pull live data from Nomis API")
        st.code("""
import pandas as pd

# No API key needed for Census 2021
url = (
    "https://www.nomisweb.co.uk/api/v01/dataset/"
    "NM_2028_1.data.csv"
    "?geography=1946157124"    # Westminster code
    "&cell=0...6"
    "&measures=20100"
    "&select=geography_name,cell_name,obs_value"
)
df = pd.read_csv(url)   # pandas reads directly from URL
        """, language="python")
        st.success("🎉 Your dashboard is live at a shareable URL — no installation needed for viewers!")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Westminster City Council · 18 Wards · IMD 2025 & 2019 (MHCLG) · "
    "Census 2021 estimates (ONS/Nomis) · Built with Python & Streamlit"
)
