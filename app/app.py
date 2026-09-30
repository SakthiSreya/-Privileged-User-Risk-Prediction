import time
import streamlit as st, pandas as pd, plotly.express as px, plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, confusion_matrix

st.set_page_config(page_title="PrivGuard", layout="wide")

@st.cache_data
def load():
    d = pd.read_csv("data/final_scores.csv", parse_dates=["day"])
    pr = pd.read_csv("data/profiles.csv").set_index("user")
    return d, pd.read_csv("models/results.csv"), pr

df, res, prof = load()
ALERT = ["High", "Critical"]
COLORS = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"}
MEANING = [
    ("USB", "Possible data exfiltration through removable media"),
    ("PCs", "Possible lateral movement or credential misuse"),
    ("after-hours", "Off-hours activity, possible account misuse"),
    ("files", "Bulk data access or collection"),
]

def meaning(reasons):
    return [m for k, m in MEANING if k in reasons] or ["No clear threat pattern"]

def profile_bar(u):
    p = prof.loc[u]
    st.info(f"**{p.employee_name}** | Role: {p.role} | Dept: {p.department} | "
            f"Unit: {p.business_unit} | Team: {p.team} | Supervisor: {p.supervisor}")

st.title("PrivGuard: Privileged User Risk Dashboard")
scope = st.sidebar.radio("Population", ["IT admins (privileged)", "All users"])
data = df[df.is_admin == 1] if scope.startswith("IT") else df

k1, k2, k3, k4 = st.columns(4)
k1.metric("Users monitored", data.user.nunique())
k2.metric("Critical alerts", int((data.tier == "Critical").sum()))
k3.metric("High alerts", int((data.tier == "High").sum()))
k4.metric("Insider days caught",
          f"{int(data[data.tier.isin(ALERT)].label.sum())} / {int(data.label.sum())}")

t1, t2, t3, t4, t5 = st.tabs(
    ["Leaderboard", "User detail", "Model comparison", "Live simulation", "Alert budget"])

with t1:
    board = data.groupby("user").agg(
        peak_risk=("risk_score", "max"),
        avg_risk=("risk_score", "mean"),
        alert_days=("tier", lambda s: int(s.isin(ALERT).sum())),
    ).round(1).sort_values("peak_risk", ascending=False).head(20)
    board = board.join(prof[["employee_name", "role", "department"]])
    c1, c2 = st.columns([3, 2])
    c1.subheader("Top 20 risky users")
    c1.dataframe(board, use_container_width=True)
    tiers = data.tier.value_counts().reindex(list(COLORS)).fillna(0).reset_index()
    tiers.columns = ["tier", "days"]
    c2.subheader("Risk tiers")
    c2.plotly_chart(px.bar(tiers, x="tier", y="days", color="tier",
                           color_discrete_map=COLORS, log_y=True), use_container_width=True)

with t2:
    users = data.groupby("user").risk_score.max().sort_values(ascending=False).index
    u = st.selectbox("Select user", users)
    profile_bar(u)
    ud = data[data.user == u].sort_values("day")
    fig = px.line(ud, x="day", y="risk_score", title=f"Risk score over time: {u}")
    alerts = ud[ud.tier.isin(ALERT)]
    fig.add_scatter(x=alerts.day, y=alerts.risk_score, mode="markers",
                    marker=dict(color="red", size=8), name="High/Critical")
    st.plotly_chart(fig, use_container_width=True)

    days = ud.day.dt.date.tolist()
    peak = ud.loc[ud.risk_score.idxmax(), "day"].date()
    day = st.selectbox("Select day", days, index=days.index(peak))
    if st.button("Predict"):
        r = ud[ud.day.dt.date == day].iloc[0]
        a, b = st.columns(2)
        g = go.Figure(go.Indicator(mode="gauge+number", value=r.risk_score,
                                   gauge={"axis": {"range": [0, 100]},
                                          "bar": {"color": COLORS[r.tier]}}))
        g.update_layout(height=250, margin=dict(t=30, b=0))
        a.plotly_chart(g, use_container_width=True)
        b.subheader(f"Tier: {r.tier}")
        if r.tier in ALERT:
            b.write("**Why flagged:** " + r.reasons)
            b.write("**Security interpretation:**")
            for m in meaning(r.reasons):
                b.write("- " + m)
        else:
            b.success("No alert. Behavior looks normal for this user.")
        truth = "known insider activity" if r.label == 1 else "normal day"
        b.caption("Ground truth: " + truth)

        feats = ["n_logons", "ah_logons", "distinct_pcs", "new_pcs",
                 "usb_connects", "n_files", "ah_files"]
        cmp = pd.DataFrame({"feature": feats,
                            "today": r[feats].astype(float).values,
                            "user normal": ud[feats].median().values})
        cmp = cmp.melt("feature", var_name="type", value_name="count")
        st.plotly_chart(px.bar(cmp, x="feature", y="count", color="type", barmode="group",
                               title="Today vs this user's normal"), use_container_width=True)

with t3:
    st.subheader("Metrics on the time-based test set")
    st.dataframe(res, use_container_width=True)
    st.plotly_chart(px.bar(res, x="model", y="f1", color="subset", barmode="group",
                           title="F1-score by model"), use_container_width=True)
    pr = go.Figure()
    for m in ["IsolationForest", "RandomForest", "XGBoost", "Ensemble"]:
        prec, rec, _ = precision_recall_curve(data.label, data[m])
        pr.add_scatter(x=rec, y=prec, name=m, mode="lines")
    pr.update_layout(title="Precision-recall curves", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(pr, use_container_width=True)
    cm = confusion_matrix(data.label, data.tier.isin(ALERT).astype(int))
    st.plotly_chart(px.imshow(cm, text_auto=True,
                              x=["Predicted normal", "Predicted alert"],
                              y=["Actual normal", "Actual insider"],
                              title="Confusion matrix (High/Critical = alert)"),
                    use_container_width=True)

with t4:
    st.write("Replays one day of user activity and raises alerts as each record arrives.")
    hot = data[data.tier.isin(ALERT)].day.dt.date.value_counts()
    day = st.selectbox("Replay day", sorted(hot.index), key="sim")
    delay = st.slider("Seconds per event", 0.0, 0.5, 0.01)
    if st.button("Start replay"):
        rows = data[data.day.dt.date == day].sample(frac=1, random_state=1)
        bar, box, log = st.progress(0), st.empty(), []
        for i, (_, r) in enumerate(rows.iterrows(), 1):
            if r.tier in ALERT:
                name = prof.loc[r.user, "employee_name"]
                log.append(f"**{r.tier.upper()}** {r.user} ({name}) "
                           f"score {r.risk_score:.0f}: {r.reasons}")
            bar.progress(i / len(rows))
            box.markdown(f"Processed {i}/{len(rows)} | Alerts: {len(log)}\n\n"
                         + "\n\n".join(log[-8:]))
            time.sleep(delay)
        st.success(f"Replay done: {len(log)} alerts raised")

with t5:
    st.write("Each day the model ranks users by risk. The analyst checks only the top few.")
    per_day = data.groupby("day").user.nunique()
    active = int(per_day.median())
    max_n = int(max(2, min(50, active // 2)))
    n = st.slider("Users the analyst can check per day", 1, max_n, min(3, max_n),
                  key=f"budget_{scope}")

    ranked = data.sort_values(["day", "risk_score"], ascending=[True, False])
    top = ranked.groupby("day").head(n)
    caught, total = int(top.label.sum()), int(data.label.sum())
    x, y, z, w = st.columns(4)
    x.metric("Insider days caught", f"{caught} / {total}")
    y.metric("Detection rate", f"{caught / max(total, 1):.0%}")
    z.metric("False alerts", len(top) - caught)
    w.metric("Share of records reviewed", f"{len(top) / len(data):.1%}")
    st.caption(f"These numbers add up every day in the test period. "
               f"About {active} users are active on a typical day.")

    st.subheader("Pick a day: who would the analyst check?")
    all_days = sorted(data.day.dt.date.unique())
    ins_days = sorted(data[data.label == 1].day.dt.date.unique())
    start = all_days.index(ins_days[0]) if ins_days else 0
    d = st.selectbox("Day", all_days, index=start, key=f"bday_{scope}")

    todays = ranked[ranked.day.dt.date == d].head(n).copy()
    todays["real status"] = todays.label.map({1: "REAL INSIDER", 0: "normal"})
    todays = todays.join(prof[["employee_name", "role"]], on="user")
    todays = todays[["user", "employee_name", "role", "risk_score", "tier", "real status"]]
    todays.index = range(1, len(todays) + 1)
    st.dataframe(todays, use_container_width=True)
    st.caption("Rank 1 is the riskiest user that day. 'REAL INSIDER' means the "
               "dataset labels this user's activity that day as an attack.")