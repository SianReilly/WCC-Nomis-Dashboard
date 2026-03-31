"""
Westminster City Council — Census 2021 & IMD Analytical Dashboard
=================================================================
Built with: Python · Streamlit · Plotly · Scikit-learn · Statsmodels · SciPy
Data:       IMD 2025 & 2019 (MHCLG) · Census 2021 estimates (ONS/Nomis)
Wards:      18 Westminster City Council wards (2022 boundaries)

requirements: streamlit pandas numpy plotly scikit-learn statsmodels scipy matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error, roc_auc_score,
                              roc_curve, silhouette_score, davies_bouldin_score)
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold, StratifiedKFold
from scipy import stats
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="WCC Census & IMD", page_icon="🏛️",
                   layout="wide", initial_sidebar_state="expanded")

# ── Colours ───────────────────────────────────────────────────────────────────
WCC_BLUE  = "#003087"
WCC_GOLD  = "#C8A84B"
TEAL      = "#028090"
CORAL     = "#E87461"
SAGE      = "#84B59F"
ECON_RED  = "#E3120B"
ECON_GREY = "#CCCCCC"
LIGHT_GRID= "#E8E8E8"
DARK_TEXT = "#1A1A1A"
MID_TEXT  = "#555555"

TOTAL_WARDS_ENGLAND = 6904

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Source+Sans+3:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
  .econ-header { border-top: 4px solid #E3120B; padding: 1.2rem 0 0.8rem; margin-bottom: 1rem; }
  .econ-header h1 { font-family: 'Libre Baskerville', serif; font-size: 1.85rem; color: #1A1A1A; margin: 0 0 0.3rem; }
  .econ-header p { color: #555; font-size: 0.9rem; margin: 0; }
  .kpi-card { background: white; border-top: 3px solid #003087; padding: 0.9rem 1.1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .kpi-card h2 { margin: 0; font-size: 1.9rem; color: #003087; font-family: 'Libre Baskerville', serif; }
  .kpi-card p  { margin: 0.15rem 0 0; font-size: 0.78rem; color: #555; }
  .kpi-card.red { border-top-color: #E3120B; } .kpi-card.red h2 { color: #E3120B; }
  .kpi-card.gold { border-top-color: #C8A84B; } .kpi-card.gold h2 { color: #C8A84B; }
  .kpi-card.teal { border-top-color: #028090; } .kpi-card.teal h2 { color: #028090; }
  .section-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #E3120B; margin: 1.1rem 0 0.15rem; }
  .section-title { font-family: 'Libre Baskerville', serif; font-size: 1.05rem; font-weight: 700;
    color: #1A1A1A; border-bottom: 1px solid #E8E8E8; padding-bottom: 0.35rem; margin-bottom: 0.6rem; }
  .insight-box { background: #F7F9FC; border-left: 3px solid #028090;
    padding: 0.65rem 0.9rem; font-size: 0.85rem; margin: 0.4rem 0; }
  .warn-box { background: #FFFBF0; border-left: 3px solid #C8A84B;
    padding: 0.65rem 0.9rem; font-size: 0.85rem; margin: 0.4rem 0; }
  .alert-box { background: #FFF5F5; border-left: 3px solid #E3120B;
    padding: 0.65rem 0.9rem; font-size: 0.85rem; margin: 0.4rem 0; }
  .method-box { background: #F0F4FA; border: 1px solid #C8D8F0;
    padding: 0.9rem 1.1rem; border-radius: 4px; margin: 0.5rem 0; font-size: 0.84rem; }
  .qa-box { background: #FFF8F0; border: 1px solid #F0D080;
    padding: 0.8rem 1rem; border-radius: 4px; margin: 0.5rem 0; font-size: 0.84rem; }
  .stat-pill { font-size: 0.8rem; font-family: 'Courier New', monospace;
    background: #EEEEEE; padding: 0.2rem 0.5rem; border-radius: 3px;
    display: inline-block; margin: 0.15rem 0.1rem; }
  .source-note { font-size: 0.73rem; color: #888; margin-top: 0.25rem; }
  .context-banner { background: #1A1A2E; color: #c8e6ff; padding: 0.6rem 1rem;
    font-size: 0.82rem; margin-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  ECONOMIST THEME HELPER
# ═════════════════════════════════════════════════════════════════════════════
def econ(fig, title="", subtitle="",
         src="MHCLG IMD 2025; ONS Census 2021",
         h=420, xgrid=False):
    ttl = ""
    if title:    ttl += f"<b style='font-family:Georgia,serif;font-size:14px'>{title}</b>"
    if subtitle: ttl += f"<br><span style='font-size:10px;color:{MID_TEXT}'>{subtitle}</span>"
    fig.update_layout(
        height=h, plot_bgcolor="#FFF", paper_bgcolor="#FFF",
        font=dict(family="Arial", size=11, color=DARK_TEXT),
        title=dict(text=ttl, x=0, xanchor="left", y=0.98, yanchor="top",
                   pad=dict(l=0, t=8)),
        xaxis=dict(showgrid=xgrid, gridcolor=LIGHT_GRID, zeroline=False,
                   showline=True, linecolor=ECON_GREY, linewidth=1,
                   tickfont=dict(size=10, color=MID_TEXT),
                   title_font=dict(size=10, color=MID_TEXT)),
        yaxis=dict(showgrid=True, gridcolor=LIGHT_GRID, zeroline=False,
                   showline=False, tickfont=dict(size=10, color=MID_TEXT),
                   title_font=dict(size=10, color=MID_TEXT)),
        margin=dict(l=10, r=10, t=75 if ttl else 25, b=52 if src else 18),
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hoverlabel=dict(bgcolor="white", font_size=11, bordercolor=ECON_GREY),
    )
    fig.add_shape(type="line", xref="paper", yref="paper",
                  x0=0, x1=1, y0=1.01, y1=1.01,
                  line=dict(color=ECON_RED, width=4), layer="above")
    if src:
        fig.add_annotation(
            text=f"<span style='font-size:9px;color:#888'>Source: {src}</span>",
            xref="paper", yref="paper", x=0, y=-0.13,
            showarrow=False, align="left", xanchor="left")
    return fig


def econ_h(fig, **kwargs):
    """Economist theme for horizontal bar charts — swap grid axes."""
    fig = econ(fig, **kwargs)
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor=LIGHT_GRID, zeroline=False,
                   showline=False, tickfont=dict(size=10, color=MID_TEXT),
                   title_font=dict(size=10, color=MID_TEXT)),
        yaxis=dict(showgrid=False, zeroline=False, showline=True,
                   linecolor=ECON_GREY, linewidth=1,
                   tickfont=dict(size=10, color=MID_TEXT),
                   title_font=dict(size=10, color=MID_TEXT)),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  DATA
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    imd = {
        "Ward": ["Church Street","Westbourne","Queen's Park","Harrow Road",
                 "Pimlico South","Pimlico North","Maida Vale","Little Venice",
                 "Hyde Park","St James's","Vincent Square","Lancaster Gate",
                 "West End","Bayswater","Knightsbridge & Belgravia",
                 "Marylebone","Abbey Road","Regent's Park"],
        "IMD 2025 Score": [46.18,39.80,36.48,34.60,28.12,24.53,23.27,21.75,
                           20.91,20.90,21.34,20.34,18.05,17.54,17.55,
                           13.96,13.66,13.46],
        "IMD 2025 Rank":  [265,490,661,800,1394,1849,2066,2337,
                           2499,2501,2411,2611,3103,3234,3233,
                           4243,4337,4404],
        "IMD 2019 Score": [41.45,35.89,32.40,27.93,21.63,17.93,19.26,17.99,
                           18.57,23.18,19.77,14.60,15.92,15.79,15.45,
                           12.47,12.17,11.34],
        "IMD 2019 Rank":  [399,658,938,1416,2333,3072,2781,3064,
                           2933,2061,2666,3972,3589,3629,3719,
                           4648,4749,5056],
        "Health Score 2019": [0.29,0.29,0.06,0.00,-0.48,-0.58,-0.68,-1.04,
                              -1.14,-0.73,-0.51,-1.26,-1.09,-0.92,-1.62,
                              -1.72,-1.61,-1.93],
    }
    df = pd.DataFrame(imd)
    df["Score Change"] = (df["IMD 2025 Score"] - df["IMD 2019 Score"]).round(2)
    df["Rank Change"]  = df["IMD 2019 Rank"] - df["IMD 2025 Rank"]
    df["Deprivation Direction"] = df["Score Change"].apply(
        lambda x: "Worsened ↑" if x > 1 else ("Improved ↓" if x < -1 else "Stable →"))

    def nat_ctx(r):
        if   r <= TOTAL_WARDS_ENGLAND * 0.10: return "Top 10% most deprived"
        elif r <= TOTAL_WARDS_ENGLAND * 0.20: return "Top 20% most deprived"
        elif r <= TOTAL_WARDS_ENGLAND * 0.40: return "Top 40% most deprived"
        elif r <= TOTAL_WARDS_ENGLAND * 0.80: return "Middle 40%"
        else:                                  return "Least deprived 20%"
    df["National Context"] = df["IMD 2025 Rank"].apply(nat_ctx)

    # Census estimates anchored to real IMD ranks (modelled — see Data Sources page)
    mn, mx = df["IMD 2025 Score"].min(), df["IMD 2025 Score"].max()
    d = (df["IMD 2025 Score"] - mn) / (mx - mn)
    n = len(df)
    np.random.seed(42)
    pop           = np.random.randint(7_500, 15_500, n)
    pct_working   = (80 - d*22 + np.random.normal(0,1.5,n)).clip(52,82).round(1)
    pct_no_qual   = ( 4 + d*18 + np.random.normal(0,1.2,n)).clip(2,26).round(1)
    pct_degree    = (70 - d*35 + np.random.normal(0,2.5,n)).clip(28,75).round(1)
    pct_social    = ( 5 + d*52 + np.random.normal(0,2.5,n)).clip(3,62).round(1)
    pct_owner     = (58 - d*38 + np.random.normal(0,2.0,n)).clip(8,62).round(1)
    pct_private   = (100 - pct_social - pct_owner).clip(5,75).round(1)
    pct_white     = (82 - d*38 + np.random.normal(0,3.0,n)).clip(30,88).round(1)
    pct_asian     = ( 5 + d*18 + np.random.normal(0,1.5,n)).clip(2,30).round(1)
    pct_black     = ( 2 + d*16 + np.random.normal(0,1.5,n)).clip(1,22).round(1)
    pct_mixed     = (100 - pct_white - pct_asian - pct_black).clip(2,18).round(1)
    pct_good_hlth = (88 - d*22 + np.random.normal(0,1.5,n)).clip(60,92).round(1)
    avg_age       = (44 -  d*8 + np.random.normal(0,1.2,n)).clip(30,48).round(1)
    pct_overcrowd = ( 2 + d*20 + np.random.normal(0,1.5,n)).clip(1,26).round(1)
    median_rooms  = (4.5 - d*2 + np.random.normal(0,0.2,n)).clip(2.0,5.5).round(1)
    pct_unemp     = ( 3 + d*10 + np.random.normal(0,0.8,n)).clip(1.5,15).round(1)
    pct_long_ill  = ( 5 + d*12 + np.random.normal(0,1.0,n)).clip(3,20).round(1)

    census = pd.DataFrame({
        "Population": pop, "Employment Rate": pct_working,
        "Unemployment %": pct_unemp, "No Qualifications %": pct_no_qual,
        "Degree Level %": pct_degree, "Social Rented %": pct_social,
        "Owner Occupied %": pct_owner, "Private Rented %": pct_private,
        "White %": pct_white, "Asian %": pct_asian,
        "Black %": pct_black, "Mixed %": pct_mixed,
        "Good Health %": pct_good_hlth, "Long-term Illness %": pct_long_ill,
        "Average Age": avg_age, "Overcrowding %": pct_overcrowd,
        "Median Rooms": median_rooms,
    })
    out = pd.concat([df.reset_index(drop=True), census.reset_index(drop=True)], axis=1)
    out["IMD Quintile"] = pd.qcut(out["IMD 2025 Score"], q=5,
        labels=["Q5 — Least deprived","Q4","Q3","Q2","Q1 — Most deprived"])
    return out

@st.cache_data
def load_age():
    b = ["0–4","5–9","10–14","15–19","20–24","25–29","30–34","35–39",
         "40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80–84","85+"]
    w = [5.1,4.2,3.8,4.5,9.2,10.8,9.5,8.1,7.2,6.3,5.5,4.8,3.9,3.2,2.8,2.1,1.5,1.5]
    l = [6.2,5.5,4.9,5.1,7.9,9.8,9.2,7.8,6.7,6.0,5.3,4.5,3.8,3.1,2.6,1.9,1.2,1.5]
    return pd.DataFrame({"Age Band": b, "Westminster %": w, "London %": l})

@st.cache_data
def load_industry():
    industries = ["Finance & Insurance","Professional/Scientific","Wholesale/Retail",
                  "Accommodation & Food","Public Admin","Education","Health & Social",
                  "Info & Communication","Arts & Entertainment","Construction","Other"]
    return pd.DataFrame({
        "Industry": industries,
        "Westminster": [14.2,16.8,9.1,8.3,6.2,5.8,7.4,10.5,4.2,3.8,13.7],
        "London":      [9.1,12.3,10.8,7.9,5.2,7.1,9.8,8.4,3.9,4.6,20.9],
    })

df   = load_data()
ages = load_age()
ind  = load_industry()
WARDS = sorted(df["Ward"].tolist())


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏛️ Westminster City Council")
    st.markdown("### Census 2021 & IMD Dashboard")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview & IMD", "📉 Deprivation Trends", "👥 Demographics",
        "🏘️ Housing & Tenure", "💼 Economy & Labour",
        "📊 Statistical Analysis", "📚 Data Sources & Quality", "🐍 How It's Built",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**🔍 Filter Wards**")
    sel_wards = st.multiselect("Select wards (blank = all 18)", WARDS, default=[],
                               placeholder="All 18 wards shown")
    active = sel_wards if sel_wards else WARDS
    dff = df[df["Ward"].isin(active)].copy()
    st.caption(f"Showing {len(dff)} of 18 wards")
    st.markdown("---")
    st.caption(
        "**IMD:** [MHCLG 2025](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025)\n\n"
        "**Census:** [ONS 2021 via Nomis](https://www.nomisweb.co.uk/sources/census_2021)\n\n"
        f"**Wards:** 18 WCC wards · {TOTAL_WARDS_ENGLAND:,} in England total")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""<div class="econ-header">
  <h1>Westminster City Council — Deprivation & Census Dashboard</h1>
  <p>18 wards · IMD 2025 & 2019 (MHCLG) · Census 2021 estimates (ONS/Nomis) · WCC 2022 boundaries</p>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1: OVERVIEW & IMD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview & IMD":
    most_dep  = df.loc[df["IMD 2025 Score"].idxmax(), "Ward"]
    least_dep = df.loc[df["IMD 2025 Score"].idxmin(), "Ward"]
    gap = df["IMD 2025 Score"].max() - df["IMD 2025 Score"].min()
    worsened = (df["Deprivation Direction"]=="Worsened ↑").sum()
    st.markdown(f"""<div class="context-banner">📌 Most deprived: <b>{most_dep}</b>
    (score {df['IMD 2025 Score'].max()}) &nbsp;|&nbsp; Least deprived: <b>{least_dep}</b>
    (score {df['IMD 2025 Score'].min()}) &nbsp;|&nbsp; Borough gap: <b>{gap:.1f} pts</b>
    &nbsp;|&nbsp; Worsened since 2019: <b>{worsened}</b>
    &nbsp;|&nbsp; England total wards: <b>{TOTAL_WARDS_ENGLAND:,}</b></div>""",
    unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    top10 = (dff["IMD 2025 Rank"] <= int(TOTAL_WARDS_ENGLAND*0.1)).sum()
    with k1: st.markdown(f'<div class="kpi-card"><h2>{len(dff)}</h2><p>Wards shown</p></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="kpi-card gold"><h2>{dff["IMD 2025 Score"].mean():.1f}</h2><p>Avg IMD 2025 Score</p></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi-card red"><h2>{dff["IMD 2025 Score"].max():.1f}</h2><p>Highest ({dff.loc[dff["IMD 2025 Score"].idxmax(),"Ward"]})</p></div>', unsafe_allow_html=True)
    with k4: st.markdown(f'<div class="kpi-card"><h2>{dff["Population"].sum():,}</h2><p>Total population</p></div>', unsafe_allow_html=True)
    with k5: st.markdown(f'<div class="kpi-card teal"><h2>{top10}</h2><p>Wards in top 10% nationally</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown('<div class="section-label">Deprivation ranking</div><div class="section-title">IMD 2025 Score by ward — most to least deprived</div>', unsafe_allow_html=True)
        s_df = dff.sort_values("IMD 2025 Score", ascending=True)
        colors = [ECON_RED if s>30 else WCC_GOLD if s>20 else TEAL for s in s_df["IMD 2025 Score"]]
        fig = go.Figure(go.Bar(
            x=s_df["IMD 2025 Score"], y=s_df["Ward"], orientation="h",
            marker_color=colors,
            customdata=s_df[["IMD 2025 Rank","National Context","Score Change"]].values,
            hovertemplate="<b>%{y}</b><br>Score: %{x:.2f}<br>Rank: %{customdata[0]}<br>Context: %{customdata[1]}<br>Change 2019→2025: %{customdata[2]:+.2f}<extra></extra>",
        ))
        fig.add_vline(x=dff["IMD 2025 Score"].mean(), line_dash="dot",
                      line_color=WCC_BLUE, line_width=1.5,
                      annotation_text=f"Avg {dff['IMD 2025 Score'].mean():.1f}",
                      annotation_position="top right",
                      annotation_font=dict(size=9, color=WCC_BLUE))
        fig = econ_h(fig, h=560)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<p class="source-note">🔴 Score >30 · 🟡 20–30 · 🩵 &lt;20 | '
                    f'Top 10% nationally = rank ≤{int(TOTAL_WARDS_ENGLAND*0.1):,} of {TOTAL_WARDS_ENGLAND:,} wards | '
                    f'<a href="https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025">IMD 2025 data ↗</a></p>',
                    unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">National context</div><div class="section-title">How Westminster wards rank in England</div>', unsafe_allow_html=True)
        ctx_order = ["Top 10% most deprived","Top 20% most deprived","Top 40% most deprived","Middle 40%","Least deprived 20%"]
        ctx = dff["National Context"].value_counts().reindex(ctx_order, fill_value=0).reset_index()
        ctx.columns = ["Context","Wards"]
        ctx = ctx[ctx["Wards"]>0]
        fig2 = px.pie(ctx, names="Context", values="Wards", hole=0.45,
                      color="Context",
                      color_discrete_map={
                          "Top 10% most deprived": ECON_RED, "Top 20% most deprived": CORAL,
                          "Top 40% most deprived": WCC_GOLD, "Middle 40%": TEAL, "Least deprived 20%": WCC_BLUE})
        fig2.update_traces(textposition="outside", textfont_size=9)
        fig2 = econ(fig2, h=290, src="")
        fig2.update_layout(legend=dict(y=-0.4, font_size=9))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f'<p class="source-note">Bands: top 10% = rank ≤{int(TOTAL_WARDS_ENGLAND*0.1):,} · '
                    f'top 20% = ≤{int(TOTAL_WARDS_ENGLAND*0.2):,} · top 40% = ≤{int(TOTAL_WARDS_ENGLAND*0.4):,} '
                    f'(of {TOTAL_WARDS_ENGLAND:,} England wards)</p>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Trend 2019–2025</div><div class="section-title">Deprivation direction</div>', unsafe_allow_html=True)
        dir_df = df["Deprivation Direction"].value_counts().reset_index()
        dir_df.columns = ["Direction","Wards"]
        fig3 = px.bar(dir_df, x="Direction", y="Wards", text="Wards",
                      color="Direction",
                      color_discrete_map={"Worsened ↑":ECON_RED,"Improved ↓":TEAL,"Stable →":WCC_GOLD})
        fig3.update_traces(textposition="outside")
        fig3 = econ(fig3, h=195, src="")
        fig3.update_layout(showlegend=False, margin=dict(t=15,b=20))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-label">Correlations</div><div class="section-title">IMD 2025 Score vs census variables — Pearson r</div>', unsafe_allow_html=True)
    c_vars = ["IMD 2025 Score","Employment Rate","No Qualifications %","Social Rented %",
              "Good Health %","Overcrowding %","Degree Level %","Unemployment %","Long-term Illness %"]
    corr_fig = px.imshow(dff[c_vars].corr(), text_auto=".2f", aspect="auto",
                         color_continuous_scale=["#003087","white","#E3120B"], zmin=-1, zmax=1)
    corr_fig = econ(corr_fig, h=360,
                    subtitle="Pearson r. Red = positive correlation; blue = negative.",
                    src="ONS Census 2021 (modelled estimates); MHCLG IMD 2025")
    st.plotly_chart(corr_fig, use_container_width=True)
    st.markdown('<div class="insight-box">💡 IMD Score correlates strongly with Social Rented % and Unemployment % (positive) and with Good Health % and Degree Level % (negative) — consistent with the IMD\'s own domain structure.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2: DEPRIVATION TRENDS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📉 Deprivation Trends":
    st.markdown("""<div class="warn-box">⚠️ <b>Comparability:</b> IMD 2019 and 2025 scores are
    not directly comparable — methodology and some indicators changed between editions.
    Treat score changes as directional signals. Rank changes are more reliable.
    See Data Sources & Quality page for full details.</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">Slope chart</div><div class="section-title">IMD score 2019 vs 2025</div>', unsafe_allow_html=True)
        fig_s = go.Figure()
        for _, row in dff.sort_values("IMD 2025 Score", ascending=False).iterrows():
            c = ECON_RED if row["Score Change"]>1 else (TEAL if row["Score Change"]<-1 else WCC_GOLD)
            w = 2.5 if abs(row["Score Change"])>3 else 1.5
            fig_s.add_trace(go.Scatter(
                x=[2019,2025], y=[row["IMD 2019 Score"],row["IMD 2025 Score"]],
                mode="lines+markers+text", name=row["Ward"],
                line=dict(color=c, width=w), marker=dict(size=6),
                text=["",row["Ward"]], textposition="middle right", textfont=dict(size=8),
                hovertemplate=f"<b>{row['Ward']}</b><br>2019: {row['IMD 2019 Score']}<br>2025: {row['IMD 2025 Score']}<br>Change: {row['Score Change']:+.2f}<extra></extra>",
            ))
        fig_s.update_layout(
            xaxis=dict(tickvals=[2019,2025], showgrid=False, zeroline=False, showline=False),
            yaxis_title="IMD Score", showlegend=False)
        fig_s = econ(fig_s, h=560, src="MHCLG IMD 2025 & 2019")
        fig_s.update_layout(margin=dict(r=130))
        st.plotly_chart(fig_s, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">Change</div><div class="section-title">Score change 2019 → 2025</div>', unsafe_allow_html=True)
        cd = dff[["Ward","Score Change","Deprivation Direction"]].sort_values("Score Change", ascending=False)
        fig_c = px.bar(cd, x="Score Change", y="Ward", orientation="h",
                       color="Deprivation Direction", text="Score Change",
                       color_discrete_map={"Worsened ↑":ECON_RED,"Improved ↓":TEAL,"Stable →":WCC_GOLD})
        fig_c.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
        fig_c.add_vline(x=0, line_color="#999", line_width=1)
        fig_c = econ_h(fig_c, h=560, src="MHCLG IMD 2025 & 2019")
        fig_c.update_layout(legend=dict(y=-0.1))
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown('<div class="section-label">Full data</div><div class="section-title">Ward-level IMD detail table</div>', unsafe_allow_html=True)
    cols = ["Ward","IMD 2019 Score","IMD 2019 Rank","IMD 2025 Score","IMD 2025 Rank",
            "Score Change","Rank Change","Deprivation Direction","National Context"]
    st.dataframe(df[cols].sort_values("IMD 2025 Score", ascending=False).reset_index(drop=True),
                 use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3: DEMOGRAPHICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👥 Demographics":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">Age</div><div class="section-title">Westminster vs London — age distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=ages["Age Band"], x=ages["Westminster %"],
                             name="Westminster", orientation="h", marker_color=WCC_BLUE))
        fig.add_trace(go.Bar(y=ages["Age Band"], x=ages["London %"],
                             name="London", orientation="h", marker_color=ECON_RED, opacity=0.55))
        fig.update_layout(barmode="group", xaxis_title="% of population",
                          legend=dict(x=0.55, y=0.05))
        fig = econ_h(fig, h=420, src="ONS Census 2021")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Westminster\'s 20–34 age band is notably higher than the London average — a professional working-age bulge. Implications for housing demand and service design.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-label">Ethnicity</div><div class="section-title">Estimated ethnicity mix by ward</div>', unsafe_allow_html=True)
        eth = dff[["Ward","White %","Asian %","Black %","Mixed %"]].melt(
            id_vars="Ward", var_name="Ethnicity", value_name="%")
        fig2 = px.bar(eth, x="Ward", y="%", color="Ethnicity", barmode="stack",
                      color_discrete_map={"White %":WCC_BLUE,"Asian %":TEAL,"Black %":CORAL,"Mixed %":SAGE})
        fig2 = econ(fig2, h=420, src="ONS Census 2021 estimates (modelled)")
        fig2.update_layout(xaxis_tickangle=-45, legend=dict(y=-0.3))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-label">Health</div><div class="section-title">Self-reported health (sorted by deprivation, highest first)</div>', unsafe_allow_html=True)
    hlth = dff.sort_values("IMD 2025 Score", ascending=False)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=hlth["Ward"], y=hlth["Good Health %"], name="Good Health %", marker_color=TEAL))
    fig3.add_trace(go.Bar(x=hlth["Ward"], y=hlth["Long-term Illness %"], name="Long-term Illness %", marker_color=ECON_RED))
    fig3.update_layout(barmode="group", yaxis_title="%", xaxis_tickangle=-45)
    fig3 = econ(fig3, h=340, src="ONS Census 2021 estimates; IMD 2019 Health domain")
    st.plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 4: HOUSING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏘️ Housing & Tenure":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">Tenure</div><div class="section-title">Tenure mix by ward (sorted by deprivation)</div>', unsafe_allow_html=True)
        ten = dff.sort_values("IMD 2025 Score", ascending=False)
        ten_m = ten[["Ward","Owner Occupied %","Social Rented %","Private Rented %"]].melt(
            id_vars="Ward", var_name="Tenure", value_name="%")
        fig = px.bar(ten_m, x="%", y="Ward", orientation="h", barmode="stack",
                     color="Tenure",
                     color_discrete_map={"Owner Occupied %":WCC_BLUE,"Social Rented %":ECON_RED,"Private Rented %":TEAL})
        fig = econ_h(fig, h=520, src="ONS Census 2021 estimates (modelled)")
        fig.update_layout(legend=dict(y=-0.12))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="section-label">Overcrowding</div><div class="section-title">Social renting vs overcrowding</div>', unsafe_allow_html=True)
        fig2 = px.scatter(dff, x="Social Rented %", y="Overcrowding %",
                          color="IMD 2025 Score", hover_name="Ward",
                          size="Population", trendline="ols",
                          color_continuous_scale=["#84B59F","#C8A84B","#E3120B"])
        fig2 = econ(fig2, h=275, src="ONS Census 2021 estimates; statsmodels OLS")
        fig2.update_layout(coloraxis_colorbar=dict(title="IMD 2025"))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="section-label">Room size</div><div class="section-title">Owner occupancy vs median rooms</div>', unsafe_allow_html=True)
        fig3 = px.scatter(dff, x="Owner Occupied %", y="Median Rooms",
                          hover_name="Ward", trendline="ols",
                          color="IMD 2025 Score",
                          color_continuous_scale=["#84B59F","#E3120B"])
        fig3 = econ(fig3, h=270, src="ONS Census 2021 estimates")
        st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="insight-box">💡 Wards with the highest IMD scores (Church Street, Westbourne) show markedly higher social renting and overcrowding — consistent with the IMD\'s Housing & Services domain.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 5: ECONOMY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💼 Economy & Labour":
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown('<div class="section-label">Employment</div><div class="section-title">Employment rate by ward</div>', unsafe_allow_html=True)
        emp = dff.sort_values("Employment Rate")
        fig = px.bar(emp, x="Ward", y="Employment Rate",
                     color="IMD 2025 Score", text="Employment Rate",
                     color_continuous_scale=["#003087","#C8A84B","#E3120B"])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig = econ(fig, h=380, src="ONS Census 2021 estimates (modelled)")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="section-label">Industry</div><div class="section-title">Westminster vs London sector mix</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Westminster", x=ind["Industry"], y=ind["Westminster"], marker_color=WCC_BLUE))
        fig2.add_trace(go.Bar(name="London", x=ind["Industry"], y=ind["London"], marker_color=ECON_RED, opacity=0.65))
        fig2.update_layout(barmode="group", yaxis_title="%", xaxis_tickangle=-45)
        fig2 = econ(fig2, h=380, src="ONS Census 2021 (WCC); GLA Economics (London)")
        st.plotly_chart(fig2, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-label">Education</div><div class="section-title">Degree attainment vs employment rate</div>', unsafe_allow_html=True)
        fig3 = px.scatter(dff, x="Degree Level %", y="Employment Rate",
                          hover_name="Ward", trendline="ols",
                          color="IMD 2025 Score",
                          color_continuous_scale=["#003087","#E3120B"],
                          size="Population", size_max=30)
        fig3 = econ(fig3, h=340, src="ONS Census 2021 estimates")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        st.markdown('<div class="section-label">Unemployment</div><div class="section-title">Unemployment vs IMD score</div>', unsafe_allow_html=True)
        fig4 = px.scatter(dff, x="IMD 2025 Score", y="Unemployment %",
                          hover_name="Ward", trendline="ols",
                          color="Deprivation Direction",
                          color_discrete_map={"Worsened ↑":ECON_RED,"Improved ↓":TEAL,"Stable →":WCC_GOLD},
                          size="Population", size_max=30)
        fig4 = econ(fig4, h=340, src="ONS Census 2021 estimates; MHCLG IMD 2025")
        st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 6: STATISTICAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Statistical Analysis":

    CENSUS_FEATS = ["Employment Rate","No Qualifications %","Social Rented %",
                    "Good Health %","Overcrowding %","Degree Level %",
                    "Owner Occupied %","Average Age","Unemployment %","Long-term Illness %"]
    BASE_FEATS = ["Employment Rate","No Qualifications %","Social Rented %","Overcrowding %"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Linear Regression", "🌲 Random Forest",
        "🔵 Clustering", "🔬 Model Validation"])

    # ── REGRESSION ──────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Linear Regression — predicting IMD 2025 Score")
        st.markdown("""<div class="method-box"><b>Methodology:</b> OLS via statsmodels.
        Estimates the linear relationship between selected census predictors (X) and IMD 2025 Score (y).
        Coefficients = expected change in IMD score per 1-unit increase in each predictor, others held constant.<br>
        <b>Caveat (n=18):</b> Statistical power is limited. P-values and CIs are indicative only.
        With small n, aim for ≤4 predictors to avoid overfitting (as a rule of thumb, n/p > 10).
        </div>""", unsafe_allow_html=True)
        sel = st.multiselect("Predictor variables", CENSUS_FEATS,
                             default=BASE_FEATS)
        if len(sel) >= 1:
            X_raw = df[sel].values
            y_raw = df["IMD 2025 Score"].values
            ols = sm.OLS(y_raw, sm.add_constant(X_raw)).fit()
            y_pred = ols.predict(sm.add_constant(X_raw))
            resid = y_raw - y_pred
            dw = durbin_watson(resid)
            sw_s, sw_p = shapiro(resid)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("R²", f"{ols.rsquared:.3f}")
            m2.metric("Adj R²", f"{ols.rsquared_adj:.3f}")
            m3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_raw,y_pred)):.2f}")
            m4.metric("F-stat p", f"{ols.f_pvalue:.4f}")
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_raw, y=y_pred, mode="markers+text",
                    text=df["Ward"], textposition="top center", textfont=dict(size=8),
                    marker=dict(color=[ECON_RED if e>0 else TEAL for e in resid],
                                size=9, line=dict(width=1,color="white")),
                    hovertemplate="<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"))
                fig.add_shape(type="line", x0=y_raw.min(), y0=y_raw.min(),
                              x1=y_raw.max(), y1=y_raw.max(),
                              line=dict(color=ECON_GREY,dash="dash"))
                fig = econ(fig, title="Actual vs Predicted",
                           subtitle="Points above line = underpredicted deprivation",
                           src="", h=380)
                fig.update_layout(xaxis_title="Actual IMD", yaxis_title="Predicted IMD")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                coef_df = pd.DataFrame({
                    "Feature": sel, "Coef": ols.params[1:],
                    "CI_low": ols.conf_int()[1:,0], "CI_high": ols.conf_int()[1:,1],
                    "p_value": ols.pvalues[1:]}).sort_values("Coef")
                fig2 = go.Figure()
                for _, r in coef_df.iterrows():
                    c = ECON_RED if r["Coef"]>0 else TEAL
                    op = 1.0 if r["p_value"]<0.05 else 0.4
                    fig2.add_trace(go.Scatter(x=[r["CI_low"],r["CI_high"]], y=[r["Feature"],r["Feature"]],
                                             mode="lines", line=dict(color=c,width=2), opacity=op, showlegend=False))
                    fig2.add_trace(go.Scatter(x=[r["Coef"]], y=[r["Feature"]], mode="markers",
                                             marker=dict(color=c,size=10), opacity=op, showlegend=False,
                                             hovertemplate=f"<b>{r['Feature']}</b><br>Coef: {r['Coef']:.3f}<br>95% CI: [{r['CI_low']:.3f}, {r['CI_high']:.3f}]<br>p={r['p_value']:.3f}<extra></extra>"))
                fig2.add_vline(x=0, line_color=ECON_GREY, line_width=1.5)
                fig2 = econ(fig2, title="Coefficients + 95% CI",
                            subtitle="Faded = not significant at p<0.05", src="", h=380)
                fig2.update_layout(xaxis_title="Coefficient")
                st.plotly_chart(fig2, use_container_width=True)

            ct = coef_df.copy()
            ct["95% CI"] = ct.apply(lambda r: f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}]", axis=1)
            ct["p-value"] = ct["p_value"].apply(lambda p: f"{p:.4f} {'✅' if p<0.05 else '⚠️'}")
            ct["Direction"] = ct["Coef"].apply(lambda c: "↑ Increases deprivation" if c>0 else "↓ Reduces deprivation")
            st.dataframe(ct[["Feature","Coef","95% CI","p-value","Direction"]].rename(columns={"Coef":"Coefficient"}),
                         use_container_width=True, hide_index=True)
            st.markdown(f"""<div class="method-box">
            <b>Diagnostics:</b>
            <span class="stat-pill">Durbin-Watson = {dw:.3f}</span>
            {'✅ No autocorrelation' if 1.5<dw<2.5 else '⚠️ Possible autocorrelation'} (note: DW is for time series; for ward cross-sectional data, spatial autocorrelation / Moran\'s I is more appropriate)<br>
            <span class="stat-pill">Shapiro-Wilk W={sw_s:.4f} p={sw_p:.4f}</span>
            {'✅ Residuals plausibly normal' if sw_p>0.05 else '⚠️ Residuals may not be normal — CIs should be treated cautiously'}<br>
            <span class="stat-pill">F-stat={ols.fvalue:.2f} p={ols.f_pvalue:.4f}</span>
            {'✅ Model significant overall' if ols.f_pvalue<0.05 else '⚠️ Model not significant at p<0.05'}
            </div>""", unsafe_allow_html=True)

    # ── RANDOM FOREST ────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Random Forest — Feature Importance")
        st.markdown("""<div class="method-box"><b>Methodology:</b>
        Ensemble of decision trees (RandomForestRegressor). Each tree trains on a bootstrap sample
        and random feature subset. Predictions averaged across all trees.<br>
        <b>Feature importance (MDI):</b> Mean decrease in impurity — how often a feature is used
        for splitting and how much variance it reduces. High = more predictive.<br>
        <b>Nuance:</b> MDI can be biased towards continuous and high-cardinality features.
        Permutation importance (in Validation tab) is a more robust alternative.
        OOB R² provides an unbiased generalisation estimate.
        </div>""", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        n_t = c1.slider("Number of trees", 10, 300, 100, 10)
        m_d = c2.slider("Max tree depth", 1, 10, 5)
        rf_feats = ["Employment Rate","No Qualifications %","Social Rented %","Good Health %",
                    "Overcrowding %","Degree Level %","Owner Occupied %","Average Age",
                    "Unemployment %","Long-term Illness %","Median Rooms"]
        X_rf = df[rf_feats].values; y_rf = df["IMD 2025 Score"].values
        rf = RandomForestRegressor(n_estimators=n_t, max_depth=m_d, random_state=42, oob_score=True)
        rf.fit(X_rf, y_rf)
        rf_pred = rf.predict(X_rf)
        m1,m2,m3 = st.columns(3)
        m1.metric("Train R²", f"{r2_score(y_rf,rf_pred):.3f}")
        m2.metric("OOB R²", f"{rf.oob_score_:.3f}", help="Unbiased; each tree predicts only its out-of-bag samples")
        m3.metric("Trees", str(n_t))
        col1, col2 = st.columns(2)
        with col1:
            imp = pd.DataFrame({"Feature":rf_feats,"Importance":rf.feature_importances_}).sort_values("Importance", ascending=True)
            colors_i = [ECON_RED if v>0.15 else WCC_GOLD if v>0.08 else TEAL for v in imp["Importance"]]
            fig = go.Figure(go.Bar(x=imp["Importance"], y=imp["Feature"], orientation="h",
                                   marker_color=colors_i,
                                   text=imp["Importance"].apply(lambda x:f"{x:.3f}"), textposition="outside"))
            fig = econ_h(fig, title="Feature Importance (MDI)",
                         subtitle="Higher = more important. Red > 0.15, gold > 0.08.",
                         h=420, src="scikit-learn RandomForestRegressor")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=y_rf, y=rf_pred, mode="markers+text",
                                      text=df["Ward"], textposition="top center", textfont=dict(size=8),
                                      marker=dict(color=WCC_BLUE, size=9, line=dict(width=1,color="white")),
                                      hovertemplate="<b>%{text}</b><br>Actual: %{x:.2f}<br>RF: %{y:.2f}<extra></extra>"))
            fig2.add_shape(type="line", x0=y_rf.min(), y0=y_rf.min(),
                           x1=y_rf.max(), y1=y_rf.max(), line=dict(color=ECON_GREY,dash="dash"))
            fig2 = econ(fig2, title="Actual vs RF Predicted",
                        subtitle=f"OOB R² = {rf.oob_score_:.3f} (unbiased estimate)", src="", h=420)
            fig2.update_layout(xaxis_title="Actual IMD", yaxis_title="RF Predicted")
            st.plotly_chart(fig2, use_container_width=True)

    # ── CLUSTERING ───────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### K-Means Clustering — Ward Typologies")
        st.markdown("""<div class="method-box"><b>Methodology:</b> K-Means minimises
        within-cluster sum of squares. All features are StandardScaled before clustering —
        essential because variables on different scales would otherwise dominate distance calculations.<br>
        <b>Nuance:</b> K-Means assumes spherical clusters of similar size. Sensitive to initialisation
        (addressed via n_init=10). With n=18, clusters of &lt;4 wards should be interpreted cautiously.
        Use the elbow/silhouette plots in the Validation tab to choose k.
        </div>""", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        n_k = c1.slider("Number of clusters", 2, 6, 3)
        cl_feats = c2.multiselect("Variables to cluster on",
            ["IMD 2025 Score","Employment Rate","Degree Level %","Social Rented %",
             "Overcrowding %","Good Health %","Average Age","Unemployment %"],
            default=["IMD 2025 Score","Employment Rate","Social Rented %","Overcrowding %"])
        if len(cl_feats) >= 2:
            X_cl = df[cl_feats].values
            X_sc = StandardScaler().fit_transform(X_cl)
            km = KMeans(n_clusters=n_k, random_state=42, n_init=10)
            labels = km.fit_predict(X_sc)
            df_cl = df.copy()
            df_cl["Cluster"] = "Type " + (labels+1).astype(str)
            sil = silhouette_score(X_sc, labels)
            dbi = davies_bouldin_score(X_sc, labels)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(df_cl, x=cl_feats[0],
                                 y=cl_feats[1] if len(cl_feats)>1 else "IMD 2025 Score",
                                 color="Cluster", hover_name="Ward", size="Population", text="Ward",
                                 color_discrete_sequence=[WCC_BLUE,ECON_RED,TEAL,SAGE,WCC_GOLD,"#6D2E46"])
                fig.update_traces(textposition="top center", textfont_size=8, selector=dict(mode="markers+text"))
                fig = econ(fig, title=f"Ward Clusters (k={n_k})",
                           subtitle=f"Silhouette score = {sil:.3f}  |  Davies-Bouldin = {dbi:.3f}",
                           src="ONS Census 2021 estimates; MHCLG IMD 2025", h=420)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Cluster profiles**")
                st.dataframe(df_cl.groupby("Cluster")[cl_feats+["Population","IMD 2025 Score"]].mean().round(1),
                             use_container_width=True)
                st.markdown("**Ward assignments**")
                wc = df_cl[["Ward","Cluster","IMD 2025 Score","Population"]].sort_values(["Cluster","IMD 2025 Score"],ascending=[True,False])
                st.dataframe(wc, use_container_width=True, height=240, hide_index=True)
            m1,m2 = st.columns(2)
            m1.metric("Silhouette Score", f"{sil:.3f}", help=">0.5 = strong; 0.25–0.5 = weak; <0.25 = may be spurious")
            m2.metric("Davies-Bouldin Index", f"{dbi:.3f}", help="Lower = better separated clusters")

    # ── MODEL VALIDATION ─────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🔬 Statistical Validation")
        st.markdown("""<div class="alert-box">⚠️ <b>Small sample (n=18):</b> All tests have low
        statistical power. Confidence intervals are wide. Results are exploratory — use alongside
        qualitative evidence and subject-matter expertise.</div>""", unsafe_allow_html=True)

        vt1, vt2, vt3, vt4 = st.tabs([
            "📐 Regression Diagnostics", "🔄 Cross-Validation",
            "📊 AUC & DeLong", "🔵 Cluster Validation"])

        X_base = df[BASE_FEATS].values; y_base = df["IMD 2025 Score"].values
        ols_base = sm.OLS(y_base, sm.add_constant(X_base)).fit()
        resid_base = y_base - ols_base.predict(sm.add_constant(X_base))

        with vt1:
            dw_b = durbin_watson(resid_base)
            sw_b, sw_pb = shapiro(resid_base)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Durbin-Watson", f"{dw_b:.3f}", help="~2 = no autocorrelation. For ward data, Moran's I is more appropriate.")
            c2.metric("Shapiro-Wilk p", f"{sw_pb:.4f}", help=">0.05 = plausibly normal residuals")
            c3.metric("F-statistic", f"{ols_base.fvalue:.2f}")
            c4.metric("F p-value", f"{ols_base.f_pvalue:.4f}")
            col1, col2 = st.columns(2)
            with col1:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=ols_base.fittedvalues, y=resid_base, mode="markers+text",
                                           text=df["Ward"], textposition="top center", textfont=dict(size=8),
                                           marker=dict(color=[ECON_RED if abs(r)>2*resid_base.std() else WCC_BLUE for r in resid_base],
                                                       size=9, line=dict(width=1,color="white")),
                                           hovertemplate="<b>%{text}</b><br>Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>"))
                fig_r.add_hline(y=0, line_color=ECON_GREY, line_dash="dash")
                fig_r.add_hline(y=2*resid_base.std(), line_color=ECON_RED, line_dash="dot", line_width=1,
                                annotation_text="±2σ", annotation_position="right")
                fig_r.add_hline(y=-2*resid_base.std(), line_color=ECON_RED, line_dash="dot", line_width=1)
                fig_r = econ(fig_r, title="Residuals vs Fitted",
                             subtitle="Red = |residual| > 2σ. Random scatter = good fit.", src="", h=360)
                fig_r.update_layout(xaxis_title="Fitted values", yaxis_title="Residuals")
                st.plotly_chart(fig_r, use_container_width=True)
            with col2:
                (osm, osr) = stats.probplot(resid_base, dist="norm")
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(x=osm[0], y=osm[1], mode="markers",
                                            marker=dict(color=WCC_BLUE,size=9), name="Residuals"))
                fig_qq.add_trace(go.Scatter(x=osm[0], y=osr[1]+osr[0]*np.array(osm[0]),
                                            mode="lines", line=dict(color=ECON_RED,dash="dash"), name="Normal line"))
                fig_qq = econ(fig_qq, title="Q-Q Plot of Residuals",
                              subtitle=f"Shapiro-Wilk W={sw_b:.4f}, p={sw_pb:.4f}. {'Plausibly normal ✅' if sw_pb>0.05 else 'Departure from normality ⚠️'}",
                              src="", h=360)
                fig_qq.update_layout(xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles", showlegend=False)
                st.plotly_chart(fig_qq, use_container_width=True)

            st.markdown("#### VIF — multicollinearity check")
            st.markdown("""<div class="method-box">VIF measures variance inflation from predictor
            correlation. VIF > 5 = moderate concern; > 10 = serious. High VIF = unreliable
            individual coefficients even if R² is high.</div>""", unsafe_allow_html=True)
            vif_df = pd.DataFrame({
                "Feature": BASE_FEATS,
                "VIF": [variance_inflation_factor(X_base, i) for i in range(X_base.shape[1])],
            })
            vif_df["Assessment"] = vif_df["VIF"].apply(lambda v: "✅ OK (<5)" if v<5 else ("⚠️ Moderate (5–10)" if v<10 else "❌ High (>10)"))
            st.dataframe(vif_df, use_container_width=True, hide_index=True)

            st.markdown("#### Cook's Distance — influential observations")
            cooks, _ = ols_base.get_influence().cooks_distance
            thresh = 4 / len(df)
            fig_ck = go.Figure(go.Bar(x=df["Ward"], y=cooks,
                                      marker_color=[ECON_RED if c>thresh else TEAL for c in cooks],
                                      hovertemplate="<b>%{x}</b><br>Cook's D: %{y:.4f}<extra></extra>"))
            fig_ck.add_hline(y=thresh, line_dash="dash", line_color=ECON_RED,
                             annotation_text=f"4/n={thresh:.3f}", annotation_position="right")
            fig_ck = econ(fig_ck, title="Cook's Distance",
                          subtitle="Above dashed line = potentially high leverage on coefficients",
                          src="", h=300)
            fig_ck.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_ck, use_container_width=True)

        with vt2:
            st.markdown("#### Cross-Validation")
            st.markdown("""<div class="method-box">
            <b>LOO-CV:</b> Most reliable for n=18. Each ward held out once as test set.
            <b>5-Fold CV:</b> ~3–4 wards per test fold — high variance between folds expected.
            <b>Note:</b> Negative CV R² means model predicts worse than using the mean — possible with very small test sets.
            </div>""", unsafe_allow_html=True)
            lr_sk = LinearRegression()
            rf_cv = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            loo = LeaveOneOut()
            kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
            loo_lr = cross_val_score(lr_sk, X_base, y_base, cv=loo, scoring="r2")
            loo_rf = cross_val_score(rf_cv, X_base, y_base, cv=loo, scoring="r2")
            kf_lr  = cross_val_score(lr_sk, X_base, y_base, cv=kf5, scoring="r2")
            kf_rf  = cross_val_score(rf_cv, X_base, y_base, cv=kf5, scoring="r2")
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("LOO R² — Linear", f"{loo_lr.mean():.3f}", delta=f"±{loo_lr.std():.3f} sd")
            m2.metric("LOO R² — RF", f"{loo_rf.mean():.3f}", delta=f"±{loo_rf.std():.3f} sd")
            m3.metric("5-Fold R² — Linear", f"{kf_lr.mean():.3f}", delta=f"±{kf_lr.std():.3f} sd")
            m4.metric("5-Fold R² — RF", f"{kf_rf.mean():.3f}", delta=f"±{kf_rf.std():.3f} sd")
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(x=list(range(1,6)), y=kf_lr, name="Linear Regression", marker_color=WCC_BLUE))
            fig_cv.add_trace(go.Bar(x=list(range(1,6)), y=kf_rf, name="Random Forest", marker_color=ECON_RED))
            fig_cv.add_hline(y=0, line_color=ECON_GREY, line_width=1)
            fig_cv = econ(fig_cv, title="5-Fold CV R² per fold",
                          subtitle="High variance between folds is expected at n=18",
                          src="scikit-learn cross_val_score", h=300)
            fig_cv.update_layout(xaxis_title="Fold", yaxis_title="R²", barmode="group")
            st.plotly_chart(fig_cv, use_container_width=True)
            fig_loo = go.Figure()
            fig_loo.add_trace(go.Scatter(x=df["Ward"], y=loo_lr, mode="markers+lines", name="LR",
                                         marker_color=WCC_BLUE, marker_size=8))
            fig_loo.add_trace(go.Scatter(x=df["Ward"], y=loo_rf, mode="markers+lines", name="RF",
                                         marker_color=ECON_RED, marker_size=8))
            fig_loo.add_hline(y=0, line_color=ECON_GREY, line_dash="dash")
            fig_loo = econ(fig_loo, title="LOO CV — R² per ward held out",
                           subtitle="Negative = model fails to predict that ward from others alone",
                           src="", h=320)
            fig_loo.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_loo, use_container_width=True)

        with vt3:
            st.markdown("#### AUC, ROC Curve & DeLong-style Bootstrap Comparison")
            st.markdown("""<div class="method-box">
            <b>Binary target:</b> IMD 2025 Score above median = "high deprivation" (1), else 0.<br>
            <b>AUC:</b> Probability model ranks a high-deprivation ward above a low-deprivation one. AUC=0.5 = random; 1.0 = perfect.<br>
            <b>DeLong test:</b> Tests whether two models have significantly different AUCs.
            Implemented here via 1,000 bootstrap resamples (formal DeLong requires correlated test-set predictions; bootstrap is appropriate for small n).<br>
            <b>5-Fold AUC:</b> Stratified k-fold gives out-of-sample AUC estimate.<br>
            <b>Model A:</b> 4 census features · <b>Model B:</b> 2 features (Employment, Degree Level %)
            </div>""", unsafe_allow_html=True)
            med = df["IMD 2025 Score"].median()
            y_bin = (df["IMD 2025 Score"] > med).astype(int)
            fa = BASE_FEATS
            fb = ["Employment Rate","Degree Level %"]
            Xa = StandardScaler().fit_transform(df[fa].values)
            Xb = StandardScaler().fit_transform(df[fb].values)
            lrA = LogisticRegression(max_iter=1000, random_state=42); lrA.fit(Xa, y_bin); pA = lrA.predict_proba(Xa)[:,1]
            lrB = LogisticRegression(max_iter=1000, random_state=42); lrB.fit(Xb, y_bin); pB = lrB.predict_proba(Xb)[:,1]
            aA = roc_auc_score(y_bin, pA); aB = roc_auc_score(y_bin, pB)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_A = cross_val_score(LogisticRegression(max_iter=1000,random_state=42), Xa, y_bin, cv=skf, scoring="roc_auc")
            cv_B = cross_val_score(LogisticRegression(max_iter=1000,random_state=42), Xb, y_bin, cv=skf, scoring="roc_auc")
            np.random.seed(99)
            diffs = []
            for _ in range(1000):
                idx = np.random.choice(len(y_bin), len(y_bin), replace=True)
                yb = y_bin.iloc[idx]
                if len(np.unique(yb)) < 2: continue
                try: diffs.append(roc_auc_score(yb,pA[idx]) - roc_auc_score(yb,pB[idx]))
                except: pass
            diffs = np.array(diffs)
            ci_lo, ci_hi = np.percentile(diffs,[2.5,97.5])
            z = diffs.mean()/(diffs.std()+1e-9)
            p_dl = 2*(1-stats.norm.cdf(abs(z)))
            col1, col2 = st.columns(2)
            with col1:
                fpA,tpA,_ = roc_curve(y_bin,pA); fpB,tpB,_ = roc_curve(y_bin,pB)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpA, y=tpA, mode="lines",
                                             name=f"Model A (4 vars) AUC={aA:.3f}", line=dict(color=WCC_BLUE,width=2.5)))
                fig_roc.add_trace(go.Scatter(x=fpB, y=tpB, mode="lines",
                                             name=f"Model B (2 vars) AUC={aB:.3f}", line=dict(color=ECON_RED,width=2.5,dash="dash")))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                             name="Random (AUC=0.5)", line=dict(color=ECON_GREY,dash="dot")))
                fig_roc = econ(fig_roc, title="ROC Curves",
                               subtitle=f"Binary: IMD > median ({med:.1f}) = high deprivation",
                               src="scikit-learn; binary threshold = median IMD", h=400)
                fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                      legend=dict(y=-0.3))
                st.plotly_chart(fig_roc, use_container_width=True)
            with col2:
                m1,m2 = st.columns(2)
                m1.metric("AUC Model A", f"{aA:.3f}"); m2.metric("AUC Model B", f"{aB:.3f}")
                m1.metric("5-Fold AUC A", f"{cv_A.mean():.3f}", delta=f"±{cv_A.std():.3f}")
                m2.metric("5-Fold AUC B", f"{cv_B.mean():.3f}", delta=f"±{cv_B.std():.3f}")
                fig_b = go.Figure(go.Histogram(x=diffs, nbinsx=40, marker_color=WCC_BLUE, opacity=0.75))
                fig_b.add_vline(x=0, line_color=ECON_RED, line_dash="dash")
                fig_b.add_vrect(x0=ci_lo, x1=ci_hi, fillcolor=WCC_GOLD, opacity=0.15,
                                annotation_text="95% CI", annotation_position="top left")
                fig_b = econ(fig_b, title="Bootstrap AUC Difference (A−B)",
                             subtitle=f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]  p={p_dl:.3f}",
                             src="1,000 bootstrap resamples", h=260)
                st.plotly_chart(fig_b, use_container_width=True)
            st.markdown(f"""<div class="method-box">
            <b>DeLong-style result:</b>
            <span class="stat-pill">AUC diff = {aA-aB:.3f}</span>
            <span class="stat-pill">95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]</span>
            <span class="stat-pill">p = {p_dl:.3f}</span><br>
            {'✅ CI excludes 0 — Model A has significantly higher AUC.' if (ci_lo>0 or ci_hi<0) else '⚠️ CI includes 0 — no significant AUC difference. Expected at n=18: wide CIs are a function of sample size, not model failure.'}<br>
            <b>Note:</b> Formal DeLong (1988) assumes specific covariance structure. Bootstrap is used here
            as an approximation appropriate for small n. Results should be confirmed with larger datasets.
            </div>""", unsafe_allow_html=True)

        with vt4:
            st.markdown("#### Clustering Validation — choosing k")
            st.markdown("""<div class="method-box">
            <b>Elbow plot:</b> WCSS decreases as k increases — look for the point where additional clusters give diminishing returns.<br>
            <b>Silhouette score:</b> How similar each ward is to its own cluster vs neighbours. Range −1 to 1; higher = better.<br>
            <b>Davies-Bouldin index:</b> Average ratio of within-cluster scatter to between-cluster separation. Lower = better.
            </div>""", unsafe_allow_html=True)
            cl_b = ["IMD 2025 Score","Employment Rate","Social Rented %","Overcrowding %"]
            X_v = StandardScaler().fit_transform(df[cl_b].values)
            k_range = range(2,9)
            wcss, sils, dbis = [], [], []
            for k in k_range:
                km_v = KMeans(n_clusters=k, random_state=42, n_init=10)
                lv = km_v.fit_predict(X_v)
                wcss.append(km_v.inertia_); sils.append(silhouette_score(X_v,lv)); dbis.append(davies_bouldin_score(X_v,lv))
            best_sil = list(k_range)[np.argmax(sils)]; best_dbi = list(k_range)[np.argmin(dbis)]
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_e = go.Figure(go.Scatter(x=list(k_range), y=wcss, mode="lines+markers",
                                             marker=dict(color=WCC_BLUE,size=8), line=dict(color=WCC_BLUE,width=2.5)))
                fig_e = econ(fig_e, title="Elbow Plot (WCSS)", subtitle="Look for the elbow", src="", h=270)
                fig_e.update_layout(xaxis_title="k", yaxis_title="WCSS")
                st.plotly_chart(fig_e, use_container_width=True)
            with col2:
                fig_s = go.Figure(go.Bar(x=list(k_range), y=sils,
                                         marker_color=[ECON_RED if k==best_sil else TEAL for k in k_range],
                                         text=[f"{s:.3f}" for s in sils], textposition="outside"))
                fig_s = econ(fig_s, title="Silhouette Score", subtitle=f"Best k={best_sil}", src="", h=270)
                fig_s.update_layout(xaxis_title="k", yaxis_title="Silhouette")
                st.plotly_chart(fig_s, use_container_width=True)
            with col3:
                fig_d = go.Figure(go.Bar(x=list(k_range), y=dbis,
                                         marker_color=[ECON_RED if k==best_dbi else TEAL for k in k_range],
                                         text=[f"{d:.3f}" for d in dbis], textposition="outside"))
                fig_d = econ(fig_d, title="Davies-Bouldin Index", subtitle=f"Best k={best_dbi} (lowest)", src="", h=270)
                fig_d.update_layout(xaxis_title="k", yaxis_title="DBI")
                st.plotly_chart(fig_d, use_container_width=True)
            st.markdown(f"""<div class="insight-box">💡 Silhouette → k={best_sil}; Davies-Bouldin → k={best_dbi}.
            {'These agree — k='+str(best_sil)+' is robust.' if best_sil==best_dbi else 'These disagree — try both and choose based on interpretability for your policy purpose.'}
            With n=18, k=3 or k=4 typically gives the most interpretable ward typologies.</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 7: DATA SOURCES & QUALITY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📚 Data Sources & Quality":
    st.markdown("## 📚 Data Sources, Methodology & Quality Assurance")
    st.markdown("""<div class="context-banner">Every data point in this dashboard is documented here —
    where it comes from, what quality issues exist, and what this means for interpretation.
    Essential reading before using any findings for policy decisions.</div>""", unsafe_allow_html=True)

    ds1, ds2, ds3 = st.tabs(["📊 IMD Data", "📋 Census 2021", "⚠️ Quality & Limitations"])

    with ds1:
        st.markdown("### Index of Multiple Deprivation (IMD)")
        st.markdown("""
**Primary source:** Ministry of Housing, Communities & Local Government (MHCLG)

🔗 [**IMD 2025 — official publication**](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025)

🔗 [**IMD 2025 Technical Report**](https://www.gov.uk/government/publications/english-indices-of-deprivation-2025-technical-report)

🔗 [**IMD 2019 — previous edition**](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)

The IMD combines 7 domains. Each domain uses multiple indicators with different reference years.
Total wards in England used for national ranking: **6,904** (ONS 2022 ward boundaries).
        """)
        st.dataframe(pd.DataFrame({
            "Domain": ["Income","Employment","Education, Skills & Training",
                       "Health Deprivation & Disability","Crime",
                       "Barriers to Housing & Services","Living Environment"],
            "Weight": ["22.5%","22.5%","13.5%","13.5%","9.3%","9.3%","9.3%"],
            "What it measures": [
                "Low-income households; benefits claimants",
                "Involuntary exclusion from work",
                "Lack of attainment and skills in the local population",
                "Risk of premature death; impaired quality of life through ill health",
                "Rates of violence, burglary, theft, criminal damage",
                "Physical and financial barriers to decent housing and local services",
                "Quality of local environment: air quality, housing condition, road accidents"
            ],
        }), use_container_width=True, hide_index=True)
        st.markdown("""<div class="warn-box">⚠️ <b>Comparability (from MHCLG Technical Report §4):</b>
        IMD 2019 and 2025 scores are NOT directly comparable. Changes include: updated data sources,
        revisions to some domain compositions, and different ward boundaries (2025 uses 2021 Census
        ward boundaries vs 2019's earlier boundaries). Score changes are directional only.
        Rank changes are more reliable for longitudinal comparison.</div>""", unsafe_allow_html=True)
        st.markdown(f"""**How national percentage bands are calculated in this dashboard:**
The rank is divided by the total number of wards in England:
`% position = rank ÷ {TOTAL_WARDS_ENGLAND:,} × 100`

| Band | Rank threshold | Calculation |
|---|---|---|
| Top 10% most deprived | ≤ {int(TOTAL_WARDS_ENGLAND*0.1):,} | {TOTAL_WARDS_ENGLAND:,} × 0.10 |
| Top 20% most deprived | ≤ {int(TOTAL_WARDS_ENGLAND*0.2):,} | {TOTAL_WARDS_ENGLAND:,} × 0.20 |
| Top 40% most deprived | ≤ {int(TOTAL_WARDS_ENGLAND*0.4):,} | {TOTAL_WARDS_ENGLAND:,} × 0.40 |
| Middle 40% | ≤ {int(TOTAL_WARDS_ENGLAND*0.8):,} | {TOTAL_WARDS_ENGLAND:,} × 0.80 |
| Least deprived 20% | > {int(TOTAL_WARDS_ENGLAND*0.8):,} | above 80th percentile |
""")

    with ds2:
        st.markdown("### Census 2021 Data")
        st.markdown("""
**Primary source:** Office for National Statistics (ONS), Census 2021, England and Wales.
Reference date: 21 March 2021.

🔗 [**All Census 2021 datasets via Nomis**](https://www.nomisweb.co.uk/sources/census_2021)

🔗 [**Census 2021 Table Finder**](https://www.nomisweb.co.uk/census/2021/data_finder)

🔗 [**ONS Quality & Methods Guide — Census 2021**](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/methodologies/qualityandmethodsguideforcensusbasedstatisticsuk2021)
        """)
        st.dataframe(pd.DataFrame({
            "Variable shown": ["Employment rate","Unemployment","Qualifications","Tenure",
                                "Ethnicity","Health","Overcrowding","Industry"],
            "Nomis dataset code": ["TS066","TS066","TS067","TS054","TS021","TS037","TS050","TS060"],
            "Geography available": ["Ward","Ward","Ward","Ward","Ward","Ward","Ward","Ward"],
            "Reference year": ["2021","2021","2021","2021","2021","2021","2021","2021"],
        }), use_container_width=True, hide_index=True)
        st.markdown("**How to access real ward-level data for Westminster via the Nomis API:**")
        st.code("""
import pandas as pd

# Example: Employment status (TS066) for Westminster wards
# No API key needed for Census 2021 data
url = (
    "https://www.nomisweb.co.uk/api/v01/dataset/"
    "NM_2066_1.data.csv"
    "?geography=TYPE298"                    # All wards in England
    "&geography_filter=PARENT:1946157124"   # Filter to Westminster LA only
    "&cell=0...9"                           # All economic activity categories
    "&measures=20100"                       # Count
    "&select=geography_name,cell_name,obs_value"
)
df = pd.read_csv(url)  # pandas reads directly from URL
df["pct"] = df["obs_value"] / df.groupby("geography_name")["obs_value"].transform("sum") * 100
        """, language="python")
        st.markdown("""<div class="alert-box">⚠️ <b>Important:</b> The census variables in this
        dashboard are <b>modelled estimates</b> statistically anchored to the real IMD ranks —
        they are not actual Census 2021 ward-level counts. They are included for illustration
        and teaching. For real policy analysis, download actual data from Nomis using the
        codes above.</div>""", unsafe_allow_html=True)

    with ds3:
        st.markdown("### ⚠️ Data Quality Issues & Limitations")
        qa = [
            ("Census: Response rate variation", "ONS Quality Guide 2021",
             "Census response rates varied across Westminster wards. Church Street and Harrow Road fell below the national average (~89%). Lower response rates increase coverage bias risk — particularly for younger males, private renters, and recent migrants. Data for these groups in high non-response wards may be less reliable.", "warn"),
            ("Census: Statistical Disclosure Control", "ONS Census 2021 Methodology",
             "ONS applied two SDC methods: (1) Targeted Record Swapping — unusual households may have records swapped with a nearby similar household. (2) Cell Key Perturbation — small counts may be adjusted by ±1–2. This means small population group counts at ward level may not be exact.", "warn"),
            ("Census: COVID-19 reference date", "ONS Census 2021",
             "All Census 2021 data reflects conditions on 21 March 2021 — during the pandemic. Employment, commuting, and health data may reflect pandemic-specific conditions rather than long-term patterns. Economic activity data in particular should be treated with care.", "warn"),
            ("Census: Small area reliability", "ONS Quality Guide 2021",
             "ONS advises caution interpreting ward-level data where any category has a count below 10. For smaller Westminster wards, some cross-tabulated variables may be suppressed or perturbed. Population estimates carry a standard error of approximately ±150–300 people.", "warn"),
            ("IMD: Temporal comparability", "MHCLG IMD 2025 Technical Report §4",
             "IMD 2025 uses data reference years from 2019–2023 depending on indicator. Income deprivation data is from 2021/22; crime from 2021–23; health from 2018–21. The index reflects conditions at different points in time, not a single consistent reference date.", "warn"),
            ("IMD: LSOA to ward aggregation", "MHCLG / ONS",
             "The official IMD is calculated at LSOA level (~1,500 population). Ward scores are derived by population-weighted LSOA aggregation. This masks within-ward variation — a ward with a moderate average score may contain LSOAs ranging from among England's most to least deprived.", "alert"),
            ("IMD: Relative measure", "MHCLG IMD 2025 Technical Report §2",
             "The IMD measures relative deprivation — how an area compares to others in England, not absolute conditions. An improving rank may mean an area improved more slowly than others, or that other areas worsened more. Both score and rank changes should be considered together.", "warn"),
            ("IMD: Health domain quality", "MHCLG IMD 2025 Technical Report §7.4",
             "The health domain uses standardised mortality ratios, hospital admissions, and GP prescribing data. Prescribing-based indicators may reflect access to and quality of GP services as much as underlying health need. Areas with better GP access may appear more deprived on this domain.", "warn"),
            ("Analysis: Ecological fallacy", "Statistical methodology",
             "Ward-level relationships do not necessarily hold for individuals. Do not infer individual behaviour or characteristics from these aggregate ward statistics.", "alert"),
            ("Analysis: Spatial autocorrelation", "Statistical methodology",
             "Adjacent wards may be more similar than distant ones. Standard regression assumes independence between observations. Spatial regression methods (spatially lagged models, spatial error models) would be more rigorous for ward-level data. Durbin-Watson tests autocorrelation in residuals but Moran's I is more appropriate for spatial data.", "warn"),
        ]
        for title, src, desc, box_type in qa:
            st.markdown(f'<div class="{box_type}-box"><b>{title}</b> <span style="color:#888;font-size:0.8rem">({src})</span><br><br>{desc}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 8: HOW IT'S BUILT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🐍 How It's Built":
    st.markdown("## 🐍 How This Dashboard Works")
    t1,t2,t3,t4 = st.tabs(["1️⃣ The Stack","2️⃣ Pandas","3️⃣ Plotly","4️⃣ Deploy"])
    with t1:
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("### 🐙 GitHub")
            st.markdown("- Version control\n- Collaboration\n- Free code hosting\n- Triggers Streamlit redeploys automatically")
        with c2:
            st.markdown("### 🐍 Python")
            st.markdown("- `pandas` — data wrangling\n- `plotly` — interactive charts\n- `scikit-learn` — ML models\n- `statsmodels` — regression diagnostics, p-values\n- `scipy` — statistical tests (Shapiro-Wilk, bootstrap)")
        with c3:
            st.markdown("### ⚡ Streamlit")
            st.markdown("- Every widget reruns the script\n- `@st.cache_data` — cache heavy loads\n- Deploy free on Streamlit Community Cloud\n- No HTML/CSS/JS needed")
    with t2:
        st.code("""
df = pd.read_csv("imd.csv")
df.describe()                           # Summary stats
df[df["IMD 2025 Score"] > 30]          # Filter
df.sort_values("IMD 2025 Score")        # Sort
df.groupby("Quintile")["Score"].mean()  # Group
df["Change"] = df["2025"] - df["2019"] # New column
        """, language="python")
    with t3:
        st.code("""
import plotly.express as px
fig = px.bar(df, x="Ward", y="IMD 2025 Score",
             color="IMD 2025 Score",
             color_continuous_scale="Reds")
fig.add_vline(x=df["IMD 2025 Score"].mean(),
              line_dash="dot", annotation_text="Avg")
st.plotly_chart(fig, use_container_width=True)
        """, language="python")
    with t4:
        st.code("""
# requirements.txt
streamlit
pandas numpy plotly
scikit-learn statsmodels scipy matplotlib

# Deploy: share.streamlit.io → New App → Connect GitHub repo
        """, language="bash")
        st.success("🎉 Dashboard live at a shareable URL — no installation needed for viewers.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Westminster City Council · 18 wards (2022 boundaries) · "
    f"[IMD 2025 (MHCLG)](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025) · "
    f"[Census 2021 (ONS/Nomis)](https://www.nomisweb.co.uk/sources/census_2021) · "
    f"National ranking based on {TOTAL_WARDS_ENGLAND:,} wards in England · "
    f"Built with Python & Streamlit"
)
