import pandas as pd, numpy as np, joblib
import xgboost as xgblib

art = joblib.load("models/privguard.pkl")
FEATURES = art["features"]
rf, xgb = art["models"]["RandomForest"], art["models"]["XGBoost"]
T = art["thresholds"]["Ensemble"]

df = pd.read_csv("data/test_scores.csv", parse_dates=["day"])
X = df[FEATURES]
b = (rf.predict_proba(X)[:, 1] + xgb.predict_proba(X)[:, 1]) / 2

# tuned threshold maps to score 50, so High/Critical == alert
score = np.where(b < T, 50 * b / T, 50 + 50 * (b - T) / (1 - T))
df["tier"] = np.select([score < 20, score < 50, score < 80],
                       ["Low", "Medium", "High"], "Critical")
df["risk_score"] = np.round(score, 1)

sv = xgb.get_booster().predict(xgblib.DMatrix(X), pred_contribs=True)[:, :-1]
top = np.argsort(-sv, axis=1)[:, :3]
names = {
    "n_logons": "logons", "ah_logons": "after-hours logons",
    "weekend_logons": "weekend logons", "distinct_pcs": "distinct PCs used",
    "new_pcs": "new PCs accessed", "usb_connects": "USB connects",
    "ah_usb": "after-hours USB use", "n_files": "files accessed",
    "ah_files": "after-hours file access", "file_pcs": "PCs used for files",
    "n_logons_z": "logon spike vs baseline", "usb_connects_z": "USB spike vs baseline",
    "n_files_z": "file access spike vs baseline", "distinct_pcs_z": "PC count spike vs baseline",
}
vals = X.to_numpy()

def explain(i):
    parts = []
    for j in top[i]:
        f, v = FEATURES[j], vals[i, j]
        if sv[i, j] > 0 and v > (0.5 if f.endswith("_z") else 0):
            parts.append(f"{names[f]} = {v:.1f}")
    return "; ".join(parts) if parts else "no strong risk signal"

df["reasons"] = [explain(i) for i in range(len(df))]
df.to_csv("data/final_scores.csv", index=False)

print("Tier counts:\n", df.tier.value_counts())
print("Insider days per tier:\n", df[df.label == 1].tier.value_counts())
print("Saved data/final_scores.csv")