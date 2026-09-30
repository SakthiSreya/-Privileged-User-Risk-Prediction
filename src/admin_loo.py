import pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("data/features.csv", parse_dates=["day"])
art = joblib.load("models/privguard.pkl")
FEATURES, T = art["features"], art["thresholds"]["Ensemble"]

insiders = sorted(df[(df.is_admin == 1) & (df.label == 1)].user.unique())
rows = []
for u in insiders:
    print("Holding out", u, "...")
    tr, te = df[df.user != u], df[df.user == u]
    neg, pos = (tr.label == 0).sum(), (tr.label == 1).sum()
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                min_samples_leaf=3, n_jobs=-1, random_state=42)
    xg = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                       scale_pos_weight=neg / max(pos, 1), eval_metric="aucpr",
                       n_jobs=-1, random_state=42)
    rf.fit(tr[FEATURES], tr.label)
    xg.fit(tr[FEATURES], tr.label)
    s = (rf.predict_proba(te[FEATURES])[:, 1] + xg.predict_proba(te[FEATURES])[:, 1]) / 2
    alert = s >= T
    a = te.label.values == 1
    rows.append({"user": u, "attack_days": int(a.sum()), "caught_days": int((alert & a).sum()),
                 "normal_days": int((~a).sum()), "false_alarms": int((alert & ~a).sum())})

res = pd.DataFrame(rows)
res["detected"] = res.caught_days > 0
print("\n", res.to_string(index=False))
print("\nInsiders detected:", int(res.detected.sum()), "of", len(res))
print("Attack-day recall:", round(res.caught_days.sum() / res.attack_days.sum(), 3))
print("False alarm rate:", round(res.false_alarms.sum() / res.normal_days.sum(), 4))
res.to_csv("models/admin_loo.csv", index=False)