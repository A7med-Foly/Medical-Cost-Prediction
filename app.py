"""
app.py  —  Medical Cost Prediction · Streamlit Dashboard
─────────────────────────────────────────────────────────
Run with:
    streamlit run app.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config (must be FIRST streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="MedCost AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --teal:    #0d9488;
    --teal-lt: #14b8a6;
    --slate:   #0f172a;
    --card:    #1e293b;
    --border:  #334155;
    --muted:   #94a3b8;
    --white:   #f8fafc;
    --warn:    #f59e0b;
    --danger:  #ef4444;
    --success: #22c55e;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Dark background */
.stApp { background: var(--slate); color: var(--white); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--white) !important; }

/* Main header */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #5eead4, #0d9488, #0891b2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin: 0;
}
.hero-sub { color: var(--muted); font-size: 1.05rem; font-weight: 300; margin-top: .4rem; }

/* Metric cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-label { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { font-size: 1.9rem; font-weight: 600; color: var(--teal-lt); line-height: 1.2; }
.metric-delta { font-size: .8rem; color: var(--muted); margin-top: .2rem; }

/* Result banner */
.result-banner {
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-label { color: rgba(255,255,255,.8); font-size: .9rem; text-transform: uppercase; letter-spacing: .1em; }
.result-amount { font-family: 'DM Serif Display', serif; font-size: 3.4rem; color: white; }
.result-sub { color: rgba(255,255,255,.7); font-size: .9rem; margin-top: .3rem; }

/* Risk badge */
.badge-low    { background:#064e3b; color:#6ee7b7; border:1px solid #065f46; border-radius:999px; padding:.25rem .9rem; font-size:.8rem; font-weight:600; display:inline-block; }
.badge-medium { background:#451a03; color:#fcd34d; border:1px solid #78350f; border-radius:999px; padding:.25rem .9rem; font-size:.8rem; font-weight:600; display:inline-block; }
.badge-high   { background:#450a0a; color:#fca5a5; border:1px solid #7f1d1d; border-radius:999px; padding:.25rem .9rem; font-size:.8rem; font-weight:600; display:inline-block; }

/* Section titles */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: var(--white);
    border-left: 3px solid var(--teal);
    padding-left: .75rem;
    margin: 1.5rem 0 1rem;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Plotly chart background */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* Streamlit elements */
div[data-testid="stSlider"] > div { color: var(--teal-lt) !important; }
div[data-baseweb="select"] { background: var(--card) !important; border-color: var(--border) !important; }
.stSelectbox label, .stSlider label, .stRadio label { color: var(--muted) !important; font-size:.85rem !important; }
div[data-testid="stMetric"] { background: var(--card); border-radius: 12px; padding: .8rem 1rem; border: 1px solid var(--border); }

button[kind="primary"] {
    background: var(--teal) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
COLORS = dict(
    teal="#0d9488", teal_lt="#14b8a6", sky="#0891b2",
    slate="#0f172a", card="#1e293b", border="#334155",
    muted="#94a3b8", warn="#f59e0b", danger="#ef4444", success="#22c55e",
)
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color=COLORS["muted"], size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=[COLORS["teal_lt"], COLORS["sky"], COLORS["warn"],
              COLORS["danger"], COLORS["success"], "#a78bfa"],
)

@st.cache_data
def load_dataset():
    """Load insurance CSV if available, otherwise generate synthetic data."""
    raw = os.path.join(os.path.dirname(__file__), "data", "raw", "insurance.csv")
    if os.path.exists(raw):
        df = pd.read_csv(raw)
    else:
        np.random.seed(42)
        n = 1338
        df = pd.DataFrame({
            "age":      np.random.randint(18, 65, n),
            "sex":      np.random.choice(["male","female"], n),
            "bmi":      np.round(np.random.uniform(16, 54, n), 2),
            "children": np.random.randint(0, 6, n),
            "smoker":   np.random.choice(["yes","no"], n, p=[0.2, 0.8]),
            "region":   np.random.choice(["southwest","southeast","northwest","northeast"], n),
        })
        df["charges"] = (
            df["age"] * 270 + df["bmi"] * 350 + df["children"] * 500
            + (df["smoker"] == "yes") * 22_000
            + np.random.normal(0, 3000, n)
        ).clip(1000)
    return df

@st.cache_resource
def load_model_artifacts():
    """Try to load trained artifacts; fall back to a quick in-memory model."""
    from config import PREPROCESSOR_PATH, BEST_MODEL_PATH
    try:
        from src.predict import load_artifacts
        preprocessor, model, poly = load_artifacts(PREPROCESSOR_PATH, BEST_MODEL_PATH)
        return preprocessor, model, poly, True
    except Exception:
        # Train a quick fallback model on synthetic data
        from sklearn.compose import make_column_transformer
        from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split

        df = load_dataset()
        X = df.drop("charges", axis=1)
        y = df["charges"]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=.2, random_state=42)

        pre = make_column_transformer(
            (MinMaxScaler(), ["age","bmi","children"]),
            (OneHotEncoder(handle_unknown="ignore"), ["sex","smoker","region"]),
        )
        pre.fit(X_train)
        model = GradientBoostingRegressor(n_estimators=200, random_state=42)
        model.fit(pre.transform(X_train), y_train)
        return pre, model, None, False

def risk_level(charge):
    if charge < 8_000:   return "Low",    "badge-low"
    if charge < 20_000:  return "Medium", "badge-medium"
    return "High", "badge-high"

def predict_charge(preprocessor, model, poly, inputs: dict) -> float:
    import pandas as pd
    row = pd.DataFrame([inputs])
    X = preprocessor.transform(row)
    if poly is not None:
        X = poly.transform(X)
    return float(model.predict(X)[0])

# ── Load data & model ──────────────────────────────────────────────────────────
df = load_dataset()
preprocessor, model, poly, model_loaded = load_model_artifacts()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedCost AI")
    st.markdown("<small style='color:#64748b'>Medical Insurance Cost Predictor</small>", unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        ["🔮 Predict", "📊 Data Explorer", "🤖 Model Insights"],
        label_visibility="collapsed",
    )
    st.divider()
    if not model_loaded:
        st.warning("⚠️ Using demo model. Run `python train.py` for full accuracy.")
    else:
        st.success("✅ Production model loaded")

    st.markdown("""
    <div style='color:#475569;font-size:.78rem;margin-top:1rem;line-height:1.6'>
    <b style='color:#64748b'>Dataset:</b> 1,338 US policyholders<br>
    <b style='color:#64748b'>Target:</b> Annual insurance charges<br>
    <b style='color:#64748b'>Best model:</b> Gradient Boosting
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict":
    st.markdown('<h1 class="hero-title">Insurance Cost Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Enter patient details to estimate annual medical insurance charges.</p>', unsafe_allow_html=True)
    st.markdown("")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-title">Patient Details</div>', unsafe_allow_html=True)

        age      = st.slider("Age", 18, 64, 35)
        bmi      = st.slider("BMI", 15.0, 54.0, 27.5, step=0.1,
                             help="Body Mass Index — weight(kg)/height(m)²")
        children = st.select_slider("Dependents", options=[0,1,2,3,4,5], value=1)

        c1, c2 = st.columns(2)
        with c1:
            sex    = st.selectbox("Biological Sex", ["male","female"])
            smoker = st.selectbox("Smoker", ["no","yes"])
        with c2:
            region = st.selectbox("Region", ["southeast","southwest","northeast","northwest"])

        st.markdown("")
        predict_btn = st.button("⚡  Estimate Cost", type="primary")

    with col_result:
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

        inputs = dict(age=age, sex=sex, bmi=bmi, children=children,
                      smoker=smoker, region=region)
        charge = predict_charge(preprocessor, model, poly, inputs)
        risk, badge_cls = risk_level(charge)

        st.markdown(f"""
        <div class="result-banner">
            <div class="result-label">Estimated Annual Charges</div>
            <div class="result-amount">${charge:,.0f}</div>
            <div class="result-sub">≈ ${charge/12:,.0f} / month</div>
        </div>
        <div style="text-align:center;margin-top:-.5rem">
            Risk Level &nbsp; <span class="{badge_cls}">{risk}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Key drivers
        st.markdown("**Key cost drivers for this profile:**")
        drivers = []
        if smoker == "yes":     drivers.append(("🚬 Smoking",       "Very High Impact", COLORS["danger"]))
        if bmi >= 30:           drivers.append(("⚖️ Obesity (BMI≥30)", "High Impact",  COLORS["warn"]))
        if age >= 50:           drivers.append(("📅 Age ≥ 50",       "Moderate Impact",  COLORS["sky"]))
        if children >= 3:       drivers.append(("👨‍👩‍👧‍👦 3+ Dependents", "Moderate Impact",COLORS["sky"]))
        if not drivers:         drivers.append(("✅ Low-risk profile","No major risk flags", COLORS["success"]))

        for label, impact, color in drivers:
            st.markdown(
                f"<div style='background:{color}22;border:1px solid {color}44;"
                f"border-radius:8px;padding:.5rem .8rem;margin:.3rem 0;"
                f"display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='color:{color};font-size:.88rem'>{label}</span>"
                f"<small style='color:{color}88'>{impact}</small></div>",
                unsafe_allow_html=True
            )

        # Comparison gauge
        avg_charge = float(df["charges"].mean())
        st.markdown("")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=charge,
            delta={"reference": avg_charge, "valueformat": ",.0f",
                   "increasing": {"color": COLORS["danger"]},
                   "decreasing": {"color": COLORS["success"]}},
            number={"prefix":"$","valueformat":",.0f",
                    "font":{"size":26,"color":COLORS["teal"]}},
            gauge={
                "axis": {"range":[0, 65000], "tickprefix":"$",
                         "tickfont":{"color":COLORS["muted"],"size":10}},
                "bar":  {"color": COLORS["teal"]},
                "bgcolor": COLORS["card"],
                "bordercolor": COLORS["border"],
                "steps": [
                    {"range":[0, 8000],  "color":"#064e3b"},
                    {"range":[8000, 20000], "color":"#451a03"},
                    {"range":[20000,65000], "color":"#450a0a"},
                ],
                "threshold": {"line":{"color":COLORS["warn"],"width":2},
                              "thickness":.75,"value":avg_charge},
            },
            title={"text":f"vs. avg ${avg_charge:,.0f}","font":{"color":COLORS["muted"],"size":12}},
        ))
    gauge.update_layout(**{**PLOT_LAYOUT, "height": 230, "margin": dict(l=20, r=20, t=30, b=10)})
    st.plotly_chart(gauge, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown('<h1 class="hero-title">Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Understand the patterns in 1,338 insurance records.</p>', unsafe_allow_html=True)

    # KPIs
    st.markdown("")
    k1, k2, k3, k4 = st.columns(4)
    stats = [
        ("Records",        f"{len(df):,}",             "US policyholders"),
        ("Avg Charge",     f"${df.charges.mean():,.0f}","per year"),
        ("Smoker Rate",    f"{(df.smoker=='yes').mean()*100:.1f}%", "of dataset"),
        ("Avg BMI",        f"{df.bmi.mean():.1f}",     "kg/m²"),
    ]
    for col, (label, value, delta) in zip([k1,k2,k3,k4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Row 1: charges dist + smoker box
    r1c1, r1c2 = st.columns(2, gap="medium")
    with r1c1:
        st.markdown('<div class="section-title">Charges Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="charges", nbins=50, color_discrete_sequence=[COLORS["teal"]])
        fig.add_vline(x=df.charges.mean(), line_dash="dash", line_color=COLORS["warn"],
                      annotation_text=f"mean ${df.charges.mean():,.0f}",
                      annotation_font_color=COLORS["warn"])
        fig.update_layout(**PLOT_LAYOUT, height=280,
                          xaxis_title="Annual Charges ($)", yaxis_title="Count")
        st.plotly_chart(fig, width='stretch')

    with r1c2:
        st.markdown('<div class="section-title">Charges by Smoker Status</div>', unsafe_allow_html=True)
        fig = px.box(df, x="smoker", y="charges", color="smoker",
                     color_discrete_map={"yes":COLORS["danger"], "no":COLORS["success"]})
        fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False,
                          xaxis_title="Smoker", yaxis_title="Annual Charges ($)")
        st.plotly_chart(fig, width='stretch')

    # Row 2: scatter + region bar
    r2c1, r2c2 = st.columns(2, gap="medium")
    with r2c1:
        st.markdown('<div class="section-title">Age vs. Charges</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x="age", y="charges", color="smoker",
                         color_discrete_map={"yes":COLORS["danger"],"no":COLORS["teal"]},
                         opacity=0.6, size_max=5)
        fig.update_layout(**PLOT_LAYOUT, height=280,
                          xaxis_title="Age", yaxis_title="Annual Charges ($)")
        st.plotly_chart(fig, width='stretch')

    with r2c2:
        st.markdown('<div class="section-title">Average Charges by Region</div>', unsafe_allow_html=True)
        region_avg = df.groupby("region")["charges"].mean().reset_index().sort_values("charges")
        fig = px.bar(region_avg, x="charges", y="region", orientation="h",
                     color="charges", color_continuous_scale=["#0d9488","#0891b2","#6366f1"])
        fig.update_layout(**PLOT_LAYOUT, height=280, coloraxis_showscale=False,
                          xaxis_title="Avg Charges ($)", yaxis_title="")
        st.plotly_chart(fig, width='stretch')

    # Row 3: BMI scatter + correlation heatmap
    r3c1, r3c2 = st.columns(2, gap="medium")
    with r3c1:
        st.markdown('<div class="section-title">BMI vs. Charges</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x="bmi", y="charges", color="smoker",
                         color_discrete_map={"yes":COLORS["danger"],"no":COLORS["teal"]},
                         opacity=0.55)
        fig.update_layout(**PLOT_LAYOUT, height=280,
                          xaxis_title="BMI", yaxis_title="Annual Charges ($)")
        st.plotly_chart(fig, width='stretch')

    with r3c2:
        st.markdown('<div class="section-title">Numeric Correlation</div>', unsafe_allow_html=True)
        corr = df[["age","bmi","children","charges"]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Teal",
                        aspect="auto")
        fig.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fig, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Insights":
    st.markdown('<h1 class="hero-title">Model Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Compare all trained models and understand what drives predictions.</p>', unsafe_allow_html=True)

    # Model comparison table (from logs or hard-coded notebook results)
    results_path = os.path.join(os.path.dirname(__file__), "logs", "model_comparison.csv")
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame({
            "Model":    ["Gradient Boosting","Polynomial Regression","Random Forest",
                         "XGBoost","KNN","Linear Regression"],
            "R2":       [0.8789, 0.8666, 0.8606, 0.8554, 0.7860, 0.7836],
            "MSE":      [18799221, 20712806, 21634332, 22454421, 33220063, 33596916],
            "RMSE":     [4335.81, 4551.13, 4651.27, 4738.61, 5763.68, 5796.28],
            "MAE":      [2411.42, 2729.50, 2564.29, 2604.33, 3599.65, 4181.19],
        })

    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    # Bar chart: R2
    fig_r2 = px.bar(
        results_df.sort_values("R2"),
        x="R2", y="Model", orientation="h",
        color="R2", color_continuous_scale=["#134e4a","#14b8a6"],
        text=results_df.sort_values("R2")["R2"].map(lambda x: f"{x:.4f}"),
    )
    fig_r2.update_traces(textposition="outside", textfont_color=COLORS["muted"])
    fig_r2.update_layout(**PLOT_LAYOUT, height=300, coloraxis_showscale=False,
                         xaxis_title="R² Score (higher = better)", yaxis_title="",
                         xaxis_range=[0.7, 0.95])
    st.plotly_chart(fig_r2, width='stretch')

    # RMSE + MAE side by side
    mc1, mc2 = st.columns(2, gap="medium")
    with mc1:
        st.markdown('<div class="section-title">RMSE by Model</div>', unsafe_allow_html=True)
        fig = px.bar(results_df.sort_values("RMSE", ascending=False),
                     x="RMSE", y="Model", orientation="h",
                     color="RMSE", color_continuous_scale=["#14b8a6","#450a0a"])
        fig.update_layout(**PLOT_LAYOUT, height=280, coloraxis_showscale=False,
                          xaxis_title="RMSE (lower = better)")
        st.plotly_chart(fig, width='stretch')

    with mc2:
        st.markdown('<div class="section-title">MAE by Model</div>', unsafe_allow_html=True)
        fig = px.bar(results_df.sort_values("MAE", ascending=False),
                     x="MAE", y="Model", orientation="h",
                     color="MAE", color_continuous_scale=["#14b8a6","#450a0a"])
        fig.update_layout(**PLOT_LAYOUT, height=280, coloraxis_showscale=False,
                          xaxis_title="MAE (lower = better)")
        st.plotly_chart(fig, width='stretch')

    # Feature importance (if GradientBoosting / RF / XGB)
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    if hasattr(model, "feature_importances_"):
        try:
            ohe_cats = preprocessor.transformers_[1][1].categories_
            cat_names = []
            for feat, cats in zip(["sex","smoker","region"], ohe_cats):
                cat_names += [f"{feat}_{c}" for c in cats]
            feat_names = ["age","bmi","children"] + cat_names
            imp = model.feature_importances_[:len(feat_names)]
            fi = pd.DataFrame({"Feature":feat_names[:len(imp)], "Importance":imp})
            fi = fi.sort_values("Importance").tail(12)
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         color="Importance",
                         color_continuous_scale=["#134e4a","#14b8a6","#0891b2"])
            fig.update_layout(**PLOT_LAYOUT, height=360, coloraxis_showscale=False,
                              xaxis_title="Feature Importance", yaxis_title="")
            st.plotly_chart(fig, width='stretch')
        except Exception:
            st.info("Feature names could not be resolved — train your model first.")
    else:
        # Show a synthetic importance chart based on known domain knowledge
        fi = pd.DataFrame({
            "Feature":    ["smoker_yes","age","bmi","region_southeast","children","sex_male"],
            "Importance": [0.62,        0.18, 0.12, 0.04,             0.025,      0.015],
        }).sort_values("Importance")
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     color="Importance",
                     color_continuous_scale=["#134e4a","#14b8a6","#0891b2"])
        fig.update_layout(**PLOT_LAYOUT, height=300, coloraxis_showscale=False,
                          xaxis_title="Feature Importance (illustrative)", yaxis_title="")
        st.plotly_chart(fig, width='stretch')
        st.caption("ℹ️ This chart shows domain-knowledge estimates. Run `python train.py` to get real importance values.")

    # Metrics table
    st.markdown('<div class="section-title">Full Metrics Table</div>', unsafe_allow_html=True)
    st.dataframe(
        results_df.style
            .bar(subset=["R2"],          color="#0d9488", vmin=0, vmax=1)
            .bar(subset=["RMSE", "MAE"], color="#ef4444")
            .format({"R2":"{:.4f}","MSE":"{:,.0f}","RMSE":"{:,.2f}","MAE":"{:,.2f}"}),
        width='stretch', hide_index=True,
    )
