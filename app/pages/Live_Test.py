import math
import streamlit as st, pandas as pd, numpy as np, joblib
import xgboost as xgblib
import plotly.graph_objects as go

st.set_page_config(page_title="PrivGuard Live Test", layout="wide")

@st.cache_resource
def load_models():
    a = joblib.load("models/privguard.pkl")
    return a["models"], a["features"], a["thresholds"]["Ensemble"]

@st.cache_data
def load_data():
    f = pd.read_csv("data/features.csv", parse_dates=["day"])
    t = pd.read_csv("data/final_scores.csv", parse_dates=["day"])
    return f, t, pd.read_csv("data/profiles.csv").set_index("user")

models, FEATURES, T = load_models()
feat, test, prof = load_data()
rf, xgb = models["RandomForest"], models["XGBoost"]
admin = test[test.is_admin == 1]

FIELDS = [
    ("n_logons", "Logons"), ("ah_logons", "After-hours logons"),
    ("weekend_logons", "Weekend logons"), ("distinct_pcs", "Distinct PCs logged into"),
    ("new_pcs", "PCs used for the first time"), ("usb_connects", "USB connects"),
    ("ah_usb", "After-hours USB connects"), ("n_files", "Files accessed"),
    ("ah_files", "After-hours file access"), ("file_pcs", "PCs used for file access"),
]
NAMES = {
    "n_logons": "logons", "ah_logons": "after-hours logons",
    "weekend_logons": "weekend logons", "distinct_pcs": "distinct PCs used",
    "new_pcs": "new PCs accessed", "usb_connects": "USB connects",
    "ah_usb": "after-hours USB use", "n_files": "files accessed",
    "ah_files": "after-hours file access", "file_pcs": "PCs used for files",
    "n_logons_z": "logon spike vs baseline", "usb_connects_z": "USB spike vs baseline",
    "n_files_z": "file access spike vs baseline", "distinct_pcs_z": "PC count spike vs baseline",
}
W = {"n_logons": 1, "ah_logons": 3, "weekend_logons": 2, "distinct_pcs": 2,
     "new_pcs": 3, "usb_connects": 3, "ah_usb": 4, "n_files": 1.5,
     "ah_files": 3, "file_pcs": 1.5}
CAPS = {"ah_logons": "n_logons", "weekend_logons": "n_logons", "ah_usb": "usb_connects",
        "ah_files": "n_files", "new_pcs": "distinct_pcs"}
MEANING = [
    ("USB", "Possible data exfiltration through removable media"),
    ("PCs", "Possible lateral movement or credential misuse"),
    ("after-hours", "Off-hours activity, possible account misuse"),
    ("files", "Bulk data access or collection"),
]
COLORS = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"}
MAXV = {k: int(max(admin[k].quantile(0.999), 10)) for k, _ in FIELDS}

def tier_of(s):
    return "Low" if s < 20 else "Medium" if s < 50 else "High" if s < 80 else "Critical"

def regular(user):
    ud = admin[(admin.user == user) & (admin.label == 0) & (admin.n_logons > 0)]
    if ud.empty:
        ud = admin[admin.user == user]
    return {k: int(round(ud[k].median())) for k, _ in FIELDS}

def ml_score(user, vals):
    h = feat[feat.user == user].tail(30)  # same 30-day baseline as training
    row = {k: float(vals[k]) for k, _ in FIELDS}
    for c in ["n_logons", "usb_connects", "n_files", "distinct_pcs"]:
        z = (row[c] - h[c].mean()) / (h[c].std() + 1)
        row[c + "_z"] = 0.0 if pd.isna(z) else z
    X = pd.DataFrame([row])[FEATURES]
    b = (rf.predict_proba(X)[0, 1] + xgb.predict_proba(X)[0, 1]) / 2
    s = 50 * b / T if b < T else 50 + 50 * (b - T) / (1 - T)
    contrib = xgb.get_booster().predict(xgblib.DMatrix(X), pred_contribs=True)[0][:-1]
    top = sorted(zip(FEATURES, contrib), key=lambda t: -t[1])[:4]
    why = [f"{NAMES[f]} = {row[f]:.0f}" for f, c in top
           if c > 0 and row[f] > (0.5 if f.endswith("_z") else 0)]
    return s, why

def baseline_score(user, vals):
    ud = admin[(admin.user == user) & (admin.label == 0)]
    contrib = {}
    for k, _ in FIELDS:
        m, s = ud[k].mean(), ud[k].std()
        s = 0 if pd.isna(s) else s
        dev = min(max((vals[k] - m) / (s + 1), 0), 5)
        contrib[k] = (W[k] * dev, dev)
    raw = sum(c for c, _ in contrib.values())
    return 100 * (1 - math.exp(-raw / 25)), contrib

def evaluate(user, vals):
    ml, ml_why = ml_score(user, vals)
    bl, contrib = baseline_score(user, vals)
    score = max(ml, bl)
    ranked = sorted(contrib.items(), key=lambda t: -t[1][0])
    bl_why = [f"{NAMES[k]} = {vals[k]}" for k, (c, d) in ranked if d >= 1.5][:5]
    reasons = ml_why if ml >= bl and ml_why else bl_why
    return ml, bl, score, tier_of(score), ranked, reasons

def reset():
    for k, v in regular(st.session_state["user"]).items():
        st.session_state[k] = min(v, MAXV[k])

admins = sorted(admin.user.unique())
if "user" not in st.session_state:
    st.session_state["user"] = admins[0]
    reset()

# after-hours/new counts can't exceed their totals
for k, cap in CAPS.items():
    if st.session_state.get(k, 0) > st.session_state.get(cap, 0):
        st.session_state[k] = st.session_state[cap]

st.title("Live Activity Test")
st.write("Sliders start at this admin's regular activity. Slide any value up and the model predicts instantly.")

st.selectbox("Admin account", admins, key="user", on_change=reset)
user = st.session_state["user"]
p = prof.loc[user]
st.info(f"**{p.employee_name}** | Role: {p.role} | Dept: {p.department} | "
        f"Unit: {p.business_unit} | Team: {p.team} | Supervisor: {p.supervisor}")

reg = regular(user)
left, right = st.columns(2)
with left:
    st.subheader("Today's activity")
    st.button("Reset to regular activity", on_click=reset, type="primary")
    for k, label in FIELDS:
        st.slider(f"{label}  (regular: {reg[k]})", 0, MAXV[k], key=k)

today = {k: st.session_state[k] for k, _ in FIELDS}
ml, bl, score, tier, ranked, reasons = evaluate(user, today)
reg_score = evaluate(user, reg)[2]

with right:
    st.subheader("Model verdict")
    g = go.Figure(go.Indicator(mode="gauge+number", value=score,
                               gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS[tier]}}))
    g.update_layout(height=260, margin=dict(t=30, b=0))
    st.plotly_chart(g, use_container_width=True)
    st.caption(f"Regular day: {reg_score:.1f} | Today: {score:.1f} | "
               f"ML: {ml:.1f} | Baseline: {bl:.1f} | Final = higher of the two.")

    if tier in ("High", "Critical"):
        st.error(f"ALERT: {tier} risk for {user} ({p.employee_name})")
    elif tier == "Medium":
        st.warning("Medium risk: worth a look")
    else:
        st.success("Low risk: looks normal")

    changed = [(n, today[k], reg[k]) for k, n in FIELDS if today[k] != reg[k]]
    if changed:
        st.write("**What you changed:**")
        for n, new, old in changed:
            st.write(f"- {n}: {old} to {new}")

    if tier in ("High", "Critical"):
        st.write("**Why flagged:** " + ("; ".join(reasons) if reasons else "combined pattern"))
        text = " ".join(reasons)
        st.write("**Security interpretation:**")
        for m in [m for k, m in MEANING if k in text] or ["Unusual combined behavior"]:
            st.write("- " + m)

    top = [(k, c) for k, (c, d) in ranked if c > 0][:6]
    if top:
        fig = go.Figure(go.Bar(x=[c for _, c in top][::-1],
                               y=[NAMES[k] for k, _ in top][::-1],
                               orientation="h", marker_color="#e74c3c"))
        fig.update_layout(title="Deviation from this admin's normal", height=280,
                          margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)