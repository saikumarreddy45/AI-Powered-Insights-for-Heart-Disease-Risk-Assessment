# app.py
import json, joblib, os, io, time, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from db import init_db, save_prediction, list_predictions, patient_history  # >>> NEW

st.set_page_config(page_title="Heart Disease Risk", layout="wide")
MODELS_DIR = "models"
ASSETS_DIR = "assets"
APP_VERSION = "v1"
API_URL = os.environ.get("API_URL")  # if set, calls FastAPI for predictions  # >>> NEW
DOCTOR_PIN = os.environ.get("DOCTOR_PIN", "2468")  # super simple demo auth   # >>> NEW

# ---------- Helpers ----------
def ensure_artifacts():
    ok = True
    missing = []
    for f in ["preprocessor.joblib", "model.joblib", "metrics.json"]:
        if not os.path.exists(os.path.join(MODELS_DIR, f)):
            ok = False
            missing.append(f)
    return ok, missing

def risk_bucket(p):  # 0..4 scale for the diagram  # >>> NEW
    if p < 0.20: return 0, "Very Low"
    if p < 0.40: return 1, "Low"
    if p < 0.60: return 2, "Medium"
    if p < 0.80: return 3, "High"
    return 4, "Very High"

def risk_class_color(level):  # >>> NEW
    # shades: green -> yellow -> red
    return ["#059669", "#10b981", "#f59e0b", "#f97316", "#dc2626"][level]

@st.cache_resource
def load_artifacts():
    pre = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    mdl = joblib.load(os.path.join(MODELS_DIR, "model.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
    return pre, mdl, metrics

def pdf_report(name, age, prob, level, label, entered: dict):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica-Bold", 20); c.drawString(50, y, "Heart Risk Assessment Report"); y -= 35
    c.setFont("Helvetica", 12); c.drawString(50, y, f"Name: {name}"); y -= 20
    if "age" in entered: c.drawString(50, y, f"Age: {int(entered['age'])}"); y -= 30

    c.setFont("Helvetica-Bold", 14); c.drawString(50, y, "Model Results"); y -= 22
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Risk Level (0–4): {level} — {label}"); y -= 18
    c.drawString(50, y, f"Risk Probability: {prob:.2f}"); y -= 26

    c.setFont("Helvetica-Bold", 14); c.drawString(50, y, "Entered Features"); y -= 20
    c.setFont("Helvetica", 11)
    for k, v in entered.items():
        c.drawString(60, y, f"• {k}: {v}")
        y -= 16
        if y < 80:
            c.showPage(); y = h - 50; c.setFont("Helvetica", 11)

    c.setFont("Helvetica-Bold", 14); c.drawString(50, y, "Recommended Actions"); y -= 20
    tips = [
        "Follow-up with cardiologist within 2–4 weeks",
        "Consider lipid profile, ECG, stress test based on symptoms",
        "Adopt Mediterranean-style diet; reduce saturated fats",
        "150+ minutes/week of moderate exercise (as advised)",
        "If smoker: begin cessation program immediately"
    ]
    c.setFont("Helvetica", 11)
    for t in tips:
        c.drawString(60, y, f"• {t}"); y -= 16
        if y < 80:
            c.showPage(); y = h - 50; c.setFont("Helvetica", 11)

    c.showPage(); c.save()
    buf.seek(0)
    return buf.getvalue()

# ---------- Guard & init ----------
ok, missing = ensure_artifacts()
if not ok:
    st.error(f"Missing artifacts: {', '.join(missing)}. Please run `python train_model.py` first.")
    st.stop()

preprocessor, model, metrics = load_artifacts()
init_db()  # >>> NEW

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
role = st.sidebar.selectbox("I am a", ["Patient", "Doctor"])  # >>> NEW
if role == "Doctor":
    pin_ok = st.sidebar.text_input("Enter PIN", type="password") == DOCTOR_PIN
else:
    pin_ok = True

page = st.sidebar.radio("Go to", ["Dashboard", "Manual Input", "Reports"] + (["Doctor Console"] if role=="Doctor" else []))
st.sidebar.markdown("---")
st.sidebar.caption("AI-Powered Insights for Heart Disease Risk Assessment")

# ---------- Patient / general pages ----------
if page == "Dashboard":
    st.title("Risk Prediction Dashboard")
    left, mid, right = st.columns([1.1, 1.2, 1.0])

    with left:
        st.subheader("Upload CSV")
        up = st.file_uploader("CSV with patient rows", type=["csv"])
        df = None
        if up:
            df = pd.read_csv(up)
            st.caption(f"Rows: {len(df)} · Cols: {len(df.columns)}")
            st.dataframe(df.head(), use_container_width=True)

    with mid:
        st.subheader("Live Prediction")
        prob_demo = 0.62
        if df is not None:
            drop_targets = [c for c in ["target","Risk","risk","label","HeartDisease","Outcome","outcome"] if c in df.columns]
            x = df.drop(columns=drop_targets) if drop_targets else df
            try:
                x_t = preprocessor.transform(x.iloc[[0]])
                prob_demo = float(model.predict_proba(x_t)[0, 1])
            except Exception as e:
                st.warning(f"Could not infer from CSV: {e}")

        level, label = risk_bucket(prob_demo)
        card = st.container()
        card.markdown(f"""
        <div style="background:{risk_class_color(level)};padding:18px;border-radius:14px;color:white;text-align:center;">
          <div style="font-size:16px;opacity:.9;">GET RESULT</div>
          <div style="font-size:44px;font-weight:800;line-height:1.1;">{label}</div>
          <div style="margin-top:6px;">Probability</div>
          <div style="font-size:34px;font-weight:700;">{prob_demo:.2f}</div>
          <div style="margin-top:6px;">Risk Level (0–4): <b>{level}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.subheader("Evaluation")
        sel = metrics["selected"]
        m = metrics["by_model"][sel]
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{m['accuracy']:.2f}")
        c2.metric("Precision", f"{m['precision']:.2f}")
        c1.metric("Recall", f"{m['recall']:.2f}")
        c2.metric("F1 Score", f"{m['f1']:.2f}")

    st.markdown("---")
    t1, t2 = st.tabs(["Performance Plots", "Global Importance"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("ROC Curve")
            p = os.path.join(ASSETS_DIR, "roc_curve.png")
            if os.path.exists(p):
                st.image(p, use_column_width=True)
            else:
                st.info("Train first to generate ROC plot.")
        with c2:
            st.caption("Model Comparison (AUROC)")
            p = os.path.join(ASSETS_DIR, "model_comparison.png")
            if os.path.exists(p):
                st.image(p, use_column_width=True)
            else:
                st.info("Train first to generate comparison plot.")
    with t2:
        p = os.path.join(ASSETS_DIR, "shap_summary.png")
        if os.path.exists(p):
            st.image(p, use_column_width=True)
        else:
            st.info("SHAP summary not available for this model.")

elif page == "Manual Input":
    st.title("Mobile-style Manual Prediction")
    st.caption("Enter patient values → tap **GET RESULT** → save to cloud & generate PDF.")

    # Suggested sliders to mimic the image  ---------------------------------
    defaults = {
        "age": 55, "sex": 1, "cp": 0, "trestbps": 130, "chol": 246,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
    }
    ints = {"sex","cp","fbs","restecg","exang","slope","ca","thal"}

    g = st.columns(3)
    values = {}
    rng = {
        "age": (18, 100), "trestbps": (80, 220), "chol": (100, 600), "thalach": (60, 220),
        "oldpeak": (0.0, 6.5), "ca": (0, 4), "cp": (0, 3), "slope": (0, 2), "thal": (0, 3)
    }
    for i, c in enumerate(list(defaults.keys())):
        with g[i % 3]:
            if c in ints:
                start, end = rng.get(c, (0, 3))
                values[c] = st.slider(c, min_value=int(start), max_value=int(end), value=int(defaults[c]))
            else:
                start, end = rng.get(c, (0.0, float(defaults[c]*2)))
                step = 0.1 if isinstance(defaults[c], float) else 1
                values[c] = st.slider(c, min_value=float(start), max_value=float(end), value=float(defaults[c]), step=step)

    name = st.text_input("Patient name", "John Doe")

    if st.button("GET RESULT", type="primary"):
        # Predict locally or via API ----------------------------------------
        if API_URL:
            try:
                resp = requests.post(f"{API_URL.rstrip('/')}/predict",
                                     json={"features": values, "patient_name": name}, timeout=10)
                resp.raise_for_status()
                res = resp.json()
                prob = float(res["prob"]); level = int(res["risk_level"]); label = res["risk_label"]
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()
        else:
            X = pd.DataFrame([values])
            x_t = preprocessor.transform(X)
            prob = float(model.predict_proba(x_t)[0, 1])
            level, label = risk_bucket(prob)

        st.success(f"**{label}** risk · Probability **{prob:.2f}** · Level **{level}**")

        # Feature bars
        feat_bar = pd.DataFrame({"Feature": list(values.keys()), "Value": list(values.values())})
        fig = px.bar(feat_bar.sort_values("Value", ascending=True),
                     x="Value", y="Feature", orientation="h",
                     title="Entered Feature Values")
        st.plotly_chart(fig, use_container_width=True)

        # Simple rules-based recommendations
        recos = []
        if values["chol"] > 240: recos.append("High cholesterol: consult dietician; lipid management.")
        if values["trestbps"] > 140: recos.append("Elevated BP: home monitoring and medication review.")
        if values["thalach"] < 120 and values["age"] > 60: recos.append("Low exercise capacity: consider supervised program.")
        if values["exang"] == 1: recos.append("Exercise-induced angina: schedule stress test per physician advice.")
        if not recos:
            recos.append("Maintain a balanced diet, regular exercise, and periodic checkups.")

        with st.expander("Recommendations"):
            for r in recos: st.markdown(f"- {r}")

        # Save to "cloud" (SQLite)  -----------------------------------------
        try:
            save_prediction(name, values, prob, level, label, metrics["selected"], APP_VERSION)
            st.toast("Saved to cloud storage", icon="💾")
        except Exception as e:
            st.warning(f"Could not save to DB: {e}")

        # PDF ---------------------------------------------------------------
        pdf = pdf_report(name, int(values.get("age", 0)), prob, level, label, values)
        st.download_button("Download PDF Report", data=pdf,
                           file_name=f"{name.replace(' ','_').lower()}_risk_report.pdf",
                           mime="application/pdf")

elif page == "Doctor Console":
    st.title("Doctor Console")
    if not pin_ok:
        st.error("Enter valid PIN in the sidebar to access.")
        st.stop()

    col = st.columns([1,1,1,1])
    with col[0]:
        q = st.text_input("Search by patient name")
    with col[1]:
        min_l = st.selectbox("Min level", [None,0,1,2,3,4], index=0)
    with col[2]:
        max_l = st.selectbox("Max level", [None,0,1,2,3,4], index=0)
    with col[3]:
        st.write("") ; st.write("")
        refresh = st.button("Refresh")

    rows = list_predictions(q or None, min_l if min_l is not None else None,
                            max_l if max_l is not None else None, limit=500)

    if not rows:
        st.info("No predictions yet.")
    else:
        # table
        df = pd.DataFrame([{
            "Time": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created_at"])),
            "Patient": r["patient_name"],
            "Prob": round(r["prob"], 3),
            "Level": r["risk_level"],
            "Label": r["risk_label"],
            "Model": r["model_name"]
        } for r in rows])
        st.dataframe(df, use_container_width=True, height=420)
        st.download_button("Export CSV", df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

        # quick patient history
        st.subheader("Patient history")
        pname = st.selectbox("Pick a patient", sorted(list({r["patient_name"] for r in rows})))
        if pname:
            hist = patient_history(pname, limit=100)
            if hist:
                hdf = pd.DataFrame(hist)
                hdf["Time"] = hdf["created_at"].apply(lambda t: time.strftime("%m-%d %H:%M", time.localtime(t)))
                line = px.line(hdf.sort_values("created_at"), x="Time", y="prob", markers=True, title=f"Risk over time — {pname}")
                st.plotly_chart(line, use_container_width=True)

else:  # Reports
    st.title("Reports & Figures")
    for title, fname in [
        ("ROC Curve", "roc_curve.png"),
        ("Model Performance", "model_comparison.png"),
        ("SHAP Feature Importance", "shap_summary.png"),
    ]:
        st.subheader(title)
        p = os.path.join(ASSETS_DIR, fname)
        if os.path.exists(p):
            st.image(p, use_column_width=True)
        else:
            st.info(f"{fname} not found. Train first.")
