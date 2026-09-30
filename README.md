# PrivGuard: Privileged User Risk Prediction 🛡️

A machine learning prototype that watches the daily activity of privileged users (IT admins) and gives each one a risk score, a risk level and a plain-English explanation.

Built for the hackathon problem statement MLCS-HACK-20: Privileged User Risk Prediction (Machine Learning for Cybersecurity).

## The Problem

Privileged users such as IT admins have access to almost everything in an organisation. If one of them turns malicious or has their account compromised, the damage is very large. Security teams cannot read millions of log records by hand, so they need a system that says who to check first and why.

## What PrivGuard Does

- Turns raw activity logs into one row per user per day
- Trains and compares three machine learning models
- Gives every user-day a risk score from 0 to 100 and a tier: Low, Medium, High or Critical
- Explains each alert with the behaviours that raised the score and the threat they may point to
- Shows everything in an interactive dashboard with a live what-if test page

## Dataset

- Name: CERT Insider Threat Test Dataset, release r4.2 (Carnegie Mellon University)
- Type: public and synthetic
- Size: about 330,000 user-day rows after processing, covering 1,000 users from January 2010 to May 2011
- Logs used: logon, device (USB), file access, LDAP (employee roles) and the insiders answer file
- Not used: email and web logs, because they are very large
- Kaggle download: https://www.kaggle.com/datasets/andrihjonior/cert-insider-threat-dataset-r4-2
- Official source: Carnegie Mellon University KiltHub, "Insider Threat Test Dataset"

The raw logs are not stored in this repository because of their size. See the How to Run section for where to place them.

Privileged users are the employees whose LDAP role is ITAdmin. There are 40 of them, and 12 are labelled insiders in the dataset. Only about 0.4 percent of all user-days are insider activity, so the data is highly imbalanced.

## Approach

1. Clean and load: parse timestamps and join logon, USB and file logs by user and day.
2. Feature engineering: one row per user per day with these features:
   - Logons, after-hours logons, weekend logons
   - Distinct PCs used and new PCs used
   - USB connects and after-hours USB connects
   - Files accessed and after-hours file access
   - Spike features that compare today with the user's own 30-day baseline
3. Labelling: a user-day is marked as malicious if it falls inside an attack window from the dataset's insiders file.
4. Time-based split, to avoid using future data for training:
   - Train: before October 2010
   - Validation: October to December 2010 (used to choose thresholds)
   - Test: January 2011 onwards
5. Training and comparison of three models.
6. Scoring: the Random Forest and XGBoost probabilities are averaged and scaled to 0-100, then mapped to Low, Medium, High and Critical.
7. Explanation: TreeSHAP contributions from XGBoost show which features pushed a score up. These are translated into text and mapped to threat types.

## Models

| Model | Type | Purpose |
|---|---|---|
| Isolation Forest | Unsupervised anomaly detection | Finds unusual days without using labels |
| Random Forest | Supervised classifier | Learns attack patterns, with class weights for imbalance |
| XGBoost | Supervised gradient boosting | Strong model for tabular data, with imbalance weighting |

## Results

Evaluated on the time-based test set, which the models never saw during training.

All users:

| Model | Precision | Recall | F1 | False Positive Rate |
|---|---|---|---|---|
| Isolation Forest | 0.01 | 0.18 | 0.02 | 8.8% |
| Random Forest | 0.62 | 0.79 | 0.70 | 0.2% |
| XGBoost | 0.48 | 0.83 | 0.60 | 0.4% |

Key findings:

- Random Forest gave the best balance of precision and recall.
- Alert budget: if analysts review only the top 1 percent riskiest user-days, the supervised models catch about 86 percent of insider days.
- Isolation Forest performed poorly on admins. Admin behaviour is already unusual (many PCs, many files), so an unsupervised detector flags almost every admin. Supervised learning is needed for privileged users.
- On the admin-only test set, the dashboard alert tiers caught 6 of 8 insider days with 24 false alarms among about 2,350 normal admin days.

The full metrics table, including false negative rate, detection rate and PR-AUC, is in models/results.csv and in the Model comparison tab of the dashboard.

## Dashboard

The Streamlit app has these parts:

- Leaderboard: the riskiest users with name, role and department, plus the tier distribution
- User detail: risk over time for one user, and a Predict button that shows score, tier, reasons, security interpretation and a today-versus-normal chart
- Model comparison: metrics table, F1 chart, precision-recall curves and confusion matrix
- Live simulation: replays a day of records and raises alerts as they arrive
- Alert budget: shows how many insider days are caught when analysts review only the top few users per day
- Live Test page: pick an admin, start from that admin's regular activity, then slide values up (after-hours logons, USB use, file access) and watch the risk score react instantly

The Live Test score combines the trained model with a baseline check of how far today is from that admin's own normal. The higher of the two is used, because tree models trained on subtle real attacks do not react strongly to extreme what-if values.

## Security Interpretation

| Pattern | Possible threat |
|---|---|
| USB use together with bulk file access | Data exfiltration |
| New PCs and many PCs used | Lateral movement or credential misuse |
| After-hours and weekend logons | Off-hours account misuse |
| Large spikes in file access | Bulk data collection |

## How to Run

1. Install Python 3.10 or newer and these packages: pandas, numpy, scikit-learn, xgboost, shap, streamlit, plotly and joblib.
2. Download the CERT r4.2 dataset from the Kaggle link above.
3. Place logon.csv, device.csv, file.csv and the LDAP folder in the data folder. Also place insiders.csv and scenarios.txt from the answers folder in the data folder.
4. Run these scripts in this order: src/make_profiles.py, src/features.py, src/train.py and src/score.py.
5. Start the dashboard with Streamlit using app/app.py. The Live Test page appears in the sidebar.

The repository already includes the trained model and the scored results, so the dashboard can run without redoing steps 2 to 4 once the small data files are present.

## Project Structure

- app: Streamlit dashboard (app.py) and the Live Test page (pages/Live_Test.py)
- src: data processing, feature engineering, training and scoring scripts
- models: saved model file and evaluation results
- data: small reference files (LDAP roles, insider labels, profiles, final scores)

## Limitations

- The dataset is synthetic. Real organisations behave differently.
- The admin test set contains only 8 insider days, so admin-only numbers are noisy.
- Every day inside an attack window is labelled malicious, although some of those days may look normal.
- Risk tier cutoffs were chosen by hand and not tuned.
- The live simulation replays daily records. It is not real log ingestion.
- Monitoring employees raises privacy and fairness concerns, and any real deployment needs clear policy and human review.

## Future Work

- Test on real or larger datasets such as CERT r6.2
- Add email and web activity features
- Try sequence models such as LSTM
- Tune tier thresholds on validation data
- Let analysts give feedback so the model keeps improving

## Team
- Members: Sakthi Sreya S, Dharshini S

## Acknowledgements

- Carnegie Mellon University CERT Division for the Insider Threat Test Dataset
- Problem statement: MLCS-HACK-20, Machine Learning for Cybersecurity hackathon
