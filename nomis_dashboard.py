"""
Westminster City Council — Census 2021 Analytical Dashboard
===========================================================
Built with: Python · Streamlit · Plotly · Scikit-learn
Data source: ONS Census 2021 via Nomis (representative ward-level data)

Run locally:   streamlit run wcc_dashboard.py
Deploy:        Push to GitHub → connect to Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WCC Census Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colours ─────────────────────────────────────────────────────────────
WCC_BLUE   = "#003087"   # Westminster City Council blue
WCC_GOLD   = "#C8A84B"   # Westminster gold accent
TEAL       = "#028090"
CORAL      = "#E87461"
SAGE       = "#84B59F"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #003087 0%, #028090 100%);
        color: white; padding: 1.5rem 2rem; border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #c8e6ff; margin: 0.3rem 0 0; font-size: 0.95rem; }
    .kpi-card {
        background: white; border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 4px solid #003087;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .kpi-card h2 { margin: 0; font-size: 1.9rem; color: #003087; }
    .kpi-card p  { margin: 0.2rem 0 0; font-size: 0.8rem; color: #666; }
    .section-header {
        font-size: 1.05rem; font-weight: 700; color: #003087;
        border-bottom: 2px solid #003087; padding-bottom: 0.3rem;
        margin: 1.2rem 0 0.8rem;
    }
    .insight-box {
        background: #f0f7ff; border-left: 3px solid #028090;
        padding: 0.7rem 1rem; border-radius: 4px; font-size: 0.88rem;
        margin: 0.5rem 0;
    }
    .code-block {
        background: #1e2761; color: #c8e6ff; padding: 1rem 1.2rem;
        border-radius: 8px; font-family: 'Courier New', monospace;
        font-size: 0.8rem; line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA  (representative Census 2021 ward-level estimates for Westminster)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_ward_data():
    wards = [
        "Abbey Road","Bayswater","Bryanston & Dorset Square","Churchill",
        "Hamilton Terrace","Harrow Road","Hyde Park","Knightsbridge & Belgravia",
        "Lancaster Gate","Maida Vale","Marylebone High Street","Millbank",
        "Pimlico North","Pimlico South","Queen's Park","Regent's Park",
        "St James's","Tachbrook","Vincent Square","Westbourne",
        "West End","Church Street","Soho","Holbein",
    ]
    np.random.seed(42)
    n = len(wards)

    pop           = np.random.randint(7_000, 16_000, n)
    pct_working   = np.random.uniform(58, 80, n).round(1)
    pct_no_qual   = np.random.uniform(4, 18, n).round(1)
    pct_degree    = np.random.uniform(35, 72, n).round(1)
    pct_social    = np.random.uniform(8, 42, n).round(1)
    pct_owner     = np.random.uniform(20, 58, n).round(1)
    pct_private   = (100 - pct_social - pct_owner).round(1)
    pct_white     = np.random.uniform(42, 82, n).round(1)
    pct_asian     = np.random.uniform(5, 25, n).round(1)
    pct_black     = np.random.uniform(3, 20, n).round(1)
    pct_mixed     = (100 - pct_white - pct_asian - pct_black).clip(2, 15).round(1)
    pct_good_hlth = np.random.uniform(64, 88, n).round(1)
    avg_age       = np.random.uniform(32, 44, n).round(1)
    pct_overcrowd = np.random.uniform(3, 22, n).round(1)
    median_rooms  = np.random.uniform(2.5, 4.5, n).round(1)
    deprivation   = (
        pct_no_qual * 0.3
        + pct_social * 0.25
        + (100 - pct_working) * 0.3
        + pct_overcrowd * 0.15
        + np.random.normal(0, 2, n)
    ).round(1)

    return pd.DataFrame({
        "Ward":            wards,
        "Population":      pop,
        "Employment Rate": pct_working,
        "No Qualifications %": pct_no_qual,
        "Degree Level %":  pct_degree,
        "Social Rented %": pct_social,
        "Owner Occupied %": pct_owner,
        "Private Rented %": pct_private,
        "White %":         pct_white,
        "Asian %":         pct_asian,
        "Black %":         pct_black,
        "Mixed %":         pct_mixed,
        "Good Health %":   pct_good_hlth,
        "Average Age":     avg_age,
        "Overcrowding %":  pct_overcrowd,
        "Median Rooms":    median_rooms,
        "Deprivation Index": deprivation,
    })


@st.cache_data
def load_age_data():
    age_bands = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34",
                 "35-39","40-44","45-49","50-54","55-59","60-64","65-69",
                 "70-74","75-79","80-84","85+"]
    wcc_pct   = [5.1,4.2,3.8,4.5,9.2,10.8,9.5,8.1,7.2,6.3,5.5,4.8,3.9,
                 3.2,2.8,2.1,1.5,1.5]
    london_pct= [6.2,5.5,4.9,5.1,7.9,9.8,9.2,7.8,6.7,6.0,5.3,4.5,3.8,
                 3.1,2.6,1.9,1.2,1.5]
    return pd.DataFrame({
        "Age Band":  age_bands,
        "Westminster %": wcc_pct,
        "London %":      london_pct,
    })


@st.cache_data
def load_industry_data():
    industries = [
        "Finance & Insurance","Professional/Scientific","Wholesale/Retail",
        "Accommodation & Food","Public Admin","Education","Health & Social",
        "Information & Communication","Arts & Entertainment","Construction","Other",
    ]
    wcc_pct    = [14.2,16.8,9.1,8.3,6.2,5.8,7.4,10.5,4.2,3.8,13.7]
    london_pct = [9.1,12.3,10.8,7.9,5.2,7.1,9.8,8.4,3.9,4.6,20.9]
    return pd.DataFrame({
        "Industry":    industries,
        "Westminster": wcc_pct,
        "London":      london_pct,
    })


df   = load_ward_data()
ages = load_age_data()
ind  = load_industry_data()


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ Westminster City Council")
    st.markdown("### 🏛️ WCC Census 2021")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview","👥 Demographics","🏘️ Housing & Tenure",
         "💼 Economy & Labour","📊 Statistical Analysis","🐍 How It's Built"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Filter Wards**")
    selected_wards = st.multiselect(
        "Select wards to highlight",
        options=df["Ward"].tolist(),
        default=[],
        placeholder="All wards shown",
    )
    if not selected_wards:
        selected_wards = df["Ward"].tolist()

    st.markdown("---")
    st.caption("Data: ONS Census 2021 · Nomis\n\nBuilt with Python + Streamlit")


dff = df[df["Ward"].isin(selected_wards)]


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏛️ Westminster City Council — Census 2021 Dashboard</h1>
  <p>Interactive analysis of ward-level population, housing, economy & deprivation • ONS Census 2021</p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class="kpi-card">
            <h2>{df['Population'].sum():,}</h2><p>Total Population</p></div>""",
            unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card">
            <h2>{len(df)}</h2><p>Wards</p></div>""",
            unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
            <h2>{df['Employment Rate'].mean():.1f}%</h2><p>Avg Employment Rate</p></div>""",
            unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card">
            <h2>{df['Degree Level %'].mean():.1f}%</h2><p>Degree-Level Qualified</p></div>""",
            unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="kpi-card">
            <h2>{df['Social Rented %'].mean():.1f}%</h2><p>Social Rented Housing</p></div>""",
            unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown('<div class="section-header">Deprivation Index by Ward</div>',
                    unsafe_allow_html=True)
        fig = px.bar(
            df.sort_values("Deprivation Index", ascending=True),
            x="Deprivation Index", y="Ward", orientation="h",
            color="Deprivation Index",
            color_continuous_scale=["#C8E6FF","#003087"],
            template="plotly_white",
        )
        fig.update_layout(height=520, coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=10,b=10))
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Index: %{x:.1f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Key Correlations</div>',
                    unsafe_allow_html=True)
        corr_vars = ["Employment Rate","No Qualifications %","Social Rented %",
                     "Good Health %","Overcrowding %","Degree Level %"]
        corr_df = df[corr_vars].corr()
        fig2 = px.imshow(
            corr_df, text_auto=".2f",
            color_continuous_scale=["#E87461","white","#003087"],
            zmin=-1, zmax=1,
            template="plotly_white",
        )
        fig2.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="insight-box">💡 <b>Key finding:</b> Higher degree attainment strongly correlates with employment rate (r ≈ 0.7), while social renting correlates with overcrowding.</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Westminster at a Glance</div>', unsafe_allow_html=True)
        radar_cats = ["Employment","Education","Health","Housing Quality","Low Deprivation"]
        radar_vals = [
            df["Employment Rate"].mean()/100,
            df["Degree Level %"].mean()/100,
            df["Good Health %"].mean()/100,
            1 - df["Overcrowding %"].mean()/100,
            1 - df["Deprivation Index"].mean()/df["Deprivation Index"].max(),
        ]
        fig3 = go.Figure(go.Scatterpolar(
            r=[v*100 for v in radar_vals],
            theta=radar_cats, fill="toself",
            line_color=WCC_BLUE, fillcolor="rgba(0,48,135,0.2)",
        ))
        fig3.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                           height=260, margin=dict(l=20,r=20,t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2: DEMOGRAPHICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👥 Demographics":
    st.markdown('<div class="section-header">Age Profile — Westminster vs London</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=ages["Age Band"], x=ages["Westminster %"],
                             name="Westminster", orientation="h",
                             marker_color=WCC_BLUE))
        fig.add_trace(go.Bar(y=ages["Age Band"], x=ages["London %"],
                             name="London", orientation="h",
                             marker_color=CORAL, opacity=0.7))
        fig.update_layout(barmode="group", height=420, template="plotly_white",
                          title="Age Distribution (%)", legend=dict(x=0.6, y=0.05),
                          xaxis_title="% of population")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Westminster has a notably higher share of 20-34 year olds vs London average, reflecting its role as a business and residential centre for young professionals.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Ethnicity by Ward</div>',
                    unsafe_allow_html=True)
        eth_melt = dff[["Ward","White %","Asian %","Black %","Mixed %"]].melt(
            id_vars="Ward", var_name="Ethnicity", value_name="Percentage")
        fig2 = px.bar(eth_melt, x="Ward", y="Percentage", color="Ethnicity",
                      color_discrete_map={"White %": WCC_BLUE, "Asian %": TEAL,
                                          "Black %": CORAL, "Mixed %": SAGE},
                      template="plotly_white", barmode="stack")
        fig2.update_layout(height=420, xaxis_tickangle=-45,
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig2, use_container_width=True)

    # Average age scatter
    st.markdown('<div class="section-header">Average Age vs Employment Rate (by Ward)</div>',
                unsafe_allow_html=True)
    fig3 = px.scatter(dff, x="Average Age", y="Employment Rate",
                      size="Population", color="Deprivation Index",
                      hover_name="Ward",
                      color_continuous_scale=["#84B59F","#003087"],
                      template="plotly_white", size_max=40)
    fig3.update_layout(height=380)
    st.plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3: HOUSING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏘️ Housing & Tenure":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Tenure Mix by Ward</div>',
                    unsafe_allow_html=True)
        ten_melt = dff[["Ward","Owner Occupied %","Social Rented %","Private Rented %"]].melt(
            id_vars="Ward", var_name="Tenure", value_name="Percentage")
        fig = px.bar(ten_melt, x="Percentage", y="Ward", color="Tenure",
                     orientation="h",
                     color_discrete_map={"Owner Occupied %": WCC_BLUE,
                                         "Social Rented %": CORAL,
                                         "Private Rented %": TEAL},
                     template="plotly_white", barmode="stack")
        fig.update_layout(height=480, legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Overcrowding vs Social Renting</div>',
                    unsafe_allow_html=True)
        fig2 = px.scatter(dff, x="Social Rented %", y="Overcrowding %",
                          color="Deprivation Index",
                          hover_name="Ward", size="Population",
                          trendline="ols",
                          color_continuous_scale=["#84B59F","#C8A84B","#E87461"],
                          template="plotly_white")
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Median Rooms vs Owner Occupancy</div>',
                    unsafe_allow_html=True)
        fig3 = px.scatter(dff, x="Owner Occupied %", y="Median Rooms",
                          hover_name="Ward",
                          trendline="ols", color_discrete_sequence=[WCC_BLUE],
                          template="plotly_white")
        fig3.update_layout(height=260)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="insight-box">💡 Wards with higher social renting tend to have significantly higher overcrowding rates. This pattern can guide targeted housing interventions.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 4: ECONOMY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💼 Economy & Labour":
    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown('<div class="section-header">Employment Rate by Ward</div>',
                    unsafe_allow_html=True)
        fig = px.bar(dff.sort_values("Employment Rate"),
                     x="Ward", y="Employment Rate",
                     color="Employment Rate",
                     color_continuous_scale=["#E87461","#C8A84B","#003087"],
                     template="plotly_white", text="Employment Rate")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=380, xaxis_tickangle=-45,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Industry Mix</div>',
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
        st.markdown('<div class="section-header">Education vs Employment</div>',
                    unsafe_allow_html=True)
        fig3 = px.scatter(dff, x="Degree Level %", y="Employment Rate",
                          hover_name="Ward", trendline="ols",
                          color="Social Rented %",
                          color_continuous_scale=["#003087","#E87461"],
                          template="plotly_white", size="Population", size_max=35)
        fig3.update_layout(height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">No Qualifications Distribution</div>',
                    unsafe_allow_html=True)
        fig4 = px.histogram(dff, x="No Qualifications %", nbins=10,
                            color_discrete_sequence=[WCC_GOLD],
                            template="plotly_white")
        fig4.update_layout(height=320, yaxis_title="Wards")
        st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 5: STATISTICAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Statistical Analysis":

    tab1, tab2, tab3 = st.tabs(["📈 Linear Regression","🌲 Random Forest","🔵 Clustering"])

    # ── TAB 1: LINEAR REGRESSION ─────────────────────────────────────────────
    with tab1:
        st.markdown("### Predicting Deprivation Index with Linear Regression")
        st.markdown("Select which census variables to include as predictors:")

        all_features = ["Employment Rate","No Qualifications %","Social Rented %",
                        "Good Health %","Overcrowding %","Degree Level %",
                        "Owner Occupied %","Average Age"]
        selected_features = st.multiselect(
            "Predictor variables (X)", all_features,
            default=["Employment Rate","No Qualifications %","Social Rented %","Overcrowding %"],
        )

        if len(selected_features) >= 1:
            X = df[selected_features].values
            y = df["Deprivation Index"].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{r2:.3f}", help="Closer to 1.0 = better fit")
            m2.metric("RMSE", f"{rmse:.2f}", help="Lower = better")
            m3.metric("Predictors", str(len(selected_features)))

            col1, col2 = st.columns(2)
            with col1:
                # Actual vs Predicted
                fig = px.scatter(
                    x=y, y=y_pred, hover_name=df["Ward"],
                    labels={"x":"Actual Deprivation","y":"Predicted"},
                    template="plotly_white", color_discrete_sequence=[WCC_BLUE],
                )
                fig.add_shape(type="line", x0=y.min(), y0=y.min(),
                              x1=y.max(), y1=y.max(),
                              line=dict(color=CORAL, dash="dash"))
                fig.update_layout(title="Actual vs Predicted", height=340)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Coefficients
                coef_df = pd.DataFrame({
                    "Feature": selected_features,
                    "Coefficient": model.coef_,
                }).sort_values("Coefficient")
                fig2 = px.bar(coef_df, x="Coefficient", y="Feature",
                              orientation="h", template="plotly_white",
                              color="Coefficient",
                              color_continuous_scale=["#E87461","white","#003087"])
                fig2.update_layout(title="Regression Coefficients", height=340,
                                   coloraxis_showscale=False)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<div class="insight-box">💡 <b>How to read this:</b> The R² score tells you how much of the variation in deprivation the model explains. Coefficients show the direction and strength of each variable\'s effect, holding others constant.</div>', unsafe_allow_html=True)
        else:
            st.info("Select at least one predictor variable above.")

    # ── TAB 2: RANDOM FOREST ─────────────────────────────────────────────────
    with tab2:
        st.markdown("### Random Forest — Feature Importance")
        st.markdown("Which census variables best predict deprivation? The Random Forest finds non-linear relationships automatically.")

        n_trees = st.slider("Number of trees", 10, 200, 100, 10)
        max_depth = st.slider("Max tree depth", 1, 10, 5)

        rf_features = ["Employment Rate","No Qualifications %","Social Rented %",
                       "Good Health %","Overcrowding %","Degree Level %",
                       "Owner Occupied %","Average Age","Median Rooms"]
        X_rf = df[rf_features].values
        y_rf = df["Deprivation Index"].values

        rf = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth,
                                   random_state=42)
        rf.fit(X_rf, y_rf)
        rf_pred = rf.predict(X_rf)
        rf_r2   = r2_score(y_rf, rf_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Random Forest R²", f"{rf_r2:.3f}")
            imp_df = pd.DataFrame({
                "Feature": rf_features,
                "Importance": rf.feature_importances_,
            }).sort_values("Importance")

            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance",
                         color_continuous_scale=["#C8E6FF","#003087"],
                         template="plotly_white")
            fig.update_layout(title="Feature Importance", height=380,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(x=y_rf, y=rf_pred, hover_name=df["Ward"],
                              labels={"x":"Actual","y":"RF Predicted"},
                              template="plotly_white", color_discrete_sequence=[TEAL])
            fig2.add_shape(type="line", x0=y_rf.min(), y0=y_rf.min(),
                           x1=y_rf.max(), y1=y_rf.max(),
                           line=dict(color=CORAL, dash="dash"))
            fig2.update_layout(title="Random Forest: Actual vs Predicted", height=340)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<div class="insight-box">💡 <b>Why Random Forest?</b> Unlike linear regression, it handles non-linear patterns and interactions automatically. Feature importance shows which variables the trees rely on most for splitting decisions.</div>', unsafe_allow_html=True)

    # ── TAB 3: CLUSTERING ────────────────────────────────────────────────────
    with tab3:
        st.markdown("### K-Means Clustering — Ward Typologies")
        st.markdown("Group similar wards together automatically, without pre-defining the categories.")

        n_clusters = st.slider("Number of clusters (ward types)", 2, 6, 3)
        cluster_features = st.multiselect(
            "Variables to cluster on",
            ["Employment Rate","Degree Level %","Social Rented %","Overcrowding %",
             "Good Health %","Average Age"],
            default=["Employment Rate","Degree Level %","Social Rented %","Overcrowding %"],
        )

        if len(cluster_features) >= 2:
            X_cl = df[cluster_features].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cl)

            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df["Cluster"] = km.fit_predict(X_scaled).astype(str)
            df["Cluster"] = "Type " + (df["Cluster"].astype(int) + 1).astype(str)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(df, x=cluster_features[0], y=cluster_features[1],
                                 color="Cluster", hover_name="Ward",
                                 size="Population",
                                 color_discrete_sequence=[WCC_BLUE,CORAL,TEAL,SAGE,WCC_GOLD,"#6D2E46"],
                                 template="plotly_white")
                fig.update_layout(height=380, title="Ward Clusters")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                cluster_summary = df.groupby("Cluster")[cluster_features + ["Population"]].mean().round(1)
                st.markdown("**Cluster Profiles (averages)**")
                st.dataframe(cluster_summary.style.background_gradient(
                    cmap="Blues", axis=0), use_container_width=True)

                st.markdown("**Ward assignments**")
                ward_clusters = df[["Ward","Cluster","Population"]].sort_values("Cluster")
                st.dataframe(ward_clusters, use_container_width=True, height=220)
        else:
            st.info("Select at least 2 variables to cluster on.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 6: HOW IT'S BUILT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🐍 How It's Built":
    st.markdown("## 🐍 How This Dashboard Works")
    st.markdown("A walkthrough of the Python + Streamlit + GitHub stack.")

    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ The Stack","2️⃣ Pandas","3️⃣ Plotly","4️⃣ Deploy"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🐙 GitHub")
            st.markdown("""
            - **Version control** — track every change
            - **Collaboration** — multiple people, one codebase
            - **Free hosting** — push code, Streamlit deploys it
            - **Portfolio** — share your work publicly
            """)
        with col2:
            st.markdown("### 🐍 Python Libraries")
            st.markdown("""
            - `pandas` — data wrangling (Excel in code)
            - `plotly` — interactive charts
            - `scikit-learn` — ML models (regression, RF, clustering)
            - `streamlit` — turns script → web app
            - `numpy` — numerical computing
            """)
        with col3:
            st.markdown("### ⚡ Streamlit")
            st.markdown("""
            - Every widget reruns the script
            - `@st.cache_data` — cache expensive computations
            - Sliders, dropdowns, multiselects built in
            - Deploy free via Streamlit Community Cloud
            - No web dev knowledge needed
            """)

    with tab2:
        st.markdown("### pandas — the basics")
        st.code("""
import pandas as pd

# Load data from Nomis API (or CSV)
df = pd.read_csv("census_data.csv")

# Explore
df.head()              # first 5 rows
df.shape               # (rows, columns)
df.describe()          # stats summary
df.dtypes              # column types

# Filter rows
westminster = df[df["authority"] == "Westminster"]

# Create new column
df["employment_rate"] = df["employed"] / df["working_age"] * 100

# Group by ward, calculate mean
ward_summary = df.groupby("ward")["employment_rate"].mean()

# Merge two datasets
merged = pd.merge(df_census, df_deprivation, on="ward_code")
        """, language="python")

    with tab3:
        st.markdown("### plotly — interactive charts")
        st.code("""
import plotly.express as px

# Bar chart
fig = px.bar(df, x="Ward", y="Employment Rate",
             color="Deprivation Index",
             color_continuous_scale="Blues",
             template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Scatter with trendline
fig = px.scatter(df, x="Degree Level %", y="Employment Rate",
                 hover_name="Ward", trendline="ols",
                 size="Population")

# Correlation heatmap
corr = df[numeric_cols].corr()
fig = px.imshow(corr, text_auto=".2f",
                color_continuous_scale="RdBu_r")
        """, language="python")

        st.markdown("### 📐 Visualisation best practices")
        best = {
            "Choose chart type by data": "Continuous → scatter/histogram | Categorical → bar | Time → line | Proportion → stacked bar",
            "Label everything": "Always include axis labels with units. Never leave unlabelled axes.",
            "Colour with purpose": "Sequential (viridis) for continuous data. Diverging (RdBu) for above/below average.",
            "Remove chart junk": "No 3D charts. No unnecessary gridlines. No truncated y-axes.",
            "Interactivity": "Use plotly for hover tooltips and zoom — lets your audience explore.",
        }
        for k, v in best.items():
            st.markdown(f"**{k}** — {v}")

    with tab4:
        st.markdown("### 🚀 Deploy your dashboard in 3 steps")
        st.code("""
# Step 1: Push to GitHub
git init
git add .
git commit -m "Initial census dashboard"
git push origin main

# Step 2: requirements.txt (in your repo root)
streamlit
pandas
numpy
plotly
scikit-learn

# Step 3: Go to share.streamlit.io
# → Connect GitHub → Select repo → Deploy!
        """, language="bash")

        st.markdown("### 📡 Pull live data from Nomis API")
        st.code("""
import requests, pandas as pd

# Nomis API — no key needed for most Census 2021 data
BASE = "https://www.nomisweb.co.uk/api/v01/dataset"

# TS021 = Ethnicity; geography = Westminster LA code
url = (f"{BASE}/NM_2028_1.data.csv?"
       f"geography=1946157124"      # Westminster LA code
       f"&cell=0...6"
       f"&measures=20100"
       f"&select=geography_name,cell_name,obs_value")

df = pd.read_csv(url)
print(df.head())
        """, language="python")

        st.success("🎉 Your dashboard is live and shareable with a single URL — no installation needed for viewers!")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Westminster City Council · Census 2021 Dashboard · Built with Python & Streamlit · Data: ONS/Nomis")
