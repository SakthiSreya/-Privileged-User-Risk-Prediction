import pandas as pd, glob

# ---------- helpers ----------
def load(name, cols):
    df = pd.read_csv(f"data/{name}.csv", usecols=cols)   # skip heavy 'content' column
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M:%S")
    df["day"] = df["date"].dt.normalize()
    hour = df["date"].dt.hour
    df["after_hours"] = ((hour < 7) | (hour >= 19)).astype(int)
    df["weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    return df

# ---------- logon ----------
print("Loading logon...")
lg = load("logon", ["date", "user", "pc", "activity"])
lg = lg[lg.activity == "Logon"]
logon = lg.groupby(["user", "day"]).agg(
    n_logons=("pc", "size"),
    ah_logons=("after_hours", "sum"),
    weekend_logons=("weekend", "sum"),
    distinct_pcs=("pc", "nunique"))
first_seen = lg.groupby(["user", "pc"])["day"].min().reset_index()
new_pcs = first_seen.groupby(["user", "day"]).size().rename("new_pcs")

# ---------- USB / device ----------
print("Loading device...")
dv = load("device", ["date", "user", "pc", "activity"])
dv = dv[dv.activity == "Connect"]
usb = dv.groupby(["user", "day"]).agg(
    usb_connects=("pc", "size"),
    ah_usb=("after_hours", "sum"))

# ---------- files ----------
print("Loading file (this is the big one)...")
fl = load("file", ["date", "user", "pc"])
files = fl.groupby(["user", "day"]).agg(
    n_files=("pc", "size"),
    ah_files=("after_hours", "sum"),
    file_pcs=("pc", "nunique"))

# ---------- combine into user-day table ----------
df = pd.concat([logon, new_pcs, usb, files], axis=1).fillna(0).reset_index()
df = df.sort_values(["user", "day"]).reset_index(drop=True)

# ---------- deviation from the user's own 30-day baseline ----------
for c in ["n_logons", "usb_connects", "n_files", "distinct_pcs"]:
    g = df.groupby("user")[c]
    mean = g.transform(lambda s: s.shift(1).rolling(30, min_periods=5).mean())
    std = g.transform(lambda s: s.shift(1).rolling(30, min_periods=5).std())
    df[c + "_z"] = ((df[c] - mean) / (std + 1)).fillna(0)

# ---------- privileged flag from LDAP ----------
ldap = pd.concat([pd.read_csv(f) for f in glob.glob("data/LDAP/*.csv")])
admins = set(ldap[ldap.role == "ITAdmin"].user_id)
df["is_admin"] = df["user"].isin(admins).astype(int)

# ---------- labels: user-days inside an insider attack window ----------
ins = pd.read_csv("data/insiders.csv")
ins = ins[ins.dataset.astype(str) == "4.2"].copy()
ins["start"] = pd.to_datetime(ins["start"]).dt.normalize()
ins["end"] = pd.to_datetime(ins["end"]).dt.normalize()
df["label"] = 0
for _, r in ins.iterrows():
    m = (df.user == r.user) & (df.day >= r.start) & (df.day <= r.end)
    df.loc[m, "label"] = 1

df.to_csv("data/features.csv", index=False)
print("\nSaved data/features.csv", df.shape)
print("Label counts (all users):\n", df.label.value_counts())
print("Label counts (admins only):\n", df[df.is_admin == 1].label.value_counts())
print("Date range:", df.day.min(), "to", df.day.max())