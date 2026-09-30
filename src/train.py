import pandas as pd, numpy as np, joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, confusion_matrix)

df = pd.read_csv("data/features.csv", parse_dates=["day"])
FEATURES = [c for c in df.columns if c not in ("user", "day", "label", "is_admin")]

# time-based split
train = df[df.day < "2010-10-01"]
val = df[(df.day >= "2010-10-01") & (df.day < "2011-01-01")]
test = df[df.day >= "2011-01-01"].copy()
for n, d in [("train", train), ("val", val), ("test", test)]:
    print(n, len(d), "insider rows:", d.label.sum(),
          "| admin insider rows:", d[d.is_admin == 1].label.sum())

neg, pos = (train.label == 0).sum(), (train.label == 1).sum()
models = {
    "IsolationForest": IsolationForest(n_estimators=200, contamination="auto", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                           min_samples_leaf=3, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                             scale_pos_weight=neg / max(pos, 1), eval_metric="aucpr",
                             n_jobs=-1, random_state=42),
}

def get_scores(name, m, X):
    if name == "IsolationForest":
        return -m.score_samples(X)
    return m.predict_proba(X)[:, 1]

def best_threshold(y, s):  # validation only
    best_t, best_f = None, -1
    for t in np.unique(np.quantile(s, np.linspace(0.90, 0.9995, 200))):
        f = f1_score(y, s >= t, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t

def evaluate(y, s, t):
    p = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"precision": precision_score(y, p, zero_division=0),
            "recall(detection rate)": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0),
            "FPR": fp / (fp + tn), "FNR": fn / (fn + tp),
            "PR-AUC": average_precision_score(y, s), "TP": tp, "FP": fp, "FN": fn}

rows, thresholds = [], {}
for name, m in models.items():
    print("Training", name, "...")
    if name == "IsolationForest":
        m.fit(train[FEATURES])
    else:
        m.fit(train[FEATURES], train.label)
    t = best_threshold(val.label, get_scores(name, m, val[FEATURES]))
    thresholds[name] = t
    test[name] = get_scores(name, m, test[FEATURES])
    for subset, d in [("all users", test), ("admins only", test[test.is_admin == 1])]:
        r = evaluate(d.label, d[name], t)
        r.update(model=name, subset=subset)
        rows.append(r)

# ensemble = the blend used by dashboard and live test
vX = val[FEATURES]
val_blend = (get_scores("RandomForest", models["RandomForest"], vX) +
             get_scores("XGBoost", models["XGBoost"], vX)) / 2
test["Ensemble"] = (test["RandomForest"] + test["XGBoost"]) / 2
thresholds["Ensemble"] = best_threshold(val.label, val_blend)
for subset, d in [("all users", test), ("admins only", test[test.is_admin == 1])]:
    r = evaluate(d.label, d["Ensemble"], thresholds["Ensemble"])
    r.update(model="Ensemble", subset=subset)
    rows.append(r)

res = pd.DataFrame(rows).set_index(["subset", "model"]).round(4)
print("\n", res.to_string())

res.to_csv("models/results.csv")
test.to_csv("data/test_scores.csv", index=False)
joblib.dump({"models": models, "features": FEATURES, "thresholds": thresholds},
            "models/privguard.pkl")
print("\nSaved models, results, test_scores")