import pandas as pd, glob

# 1. All LDAP months combined
ldap = pd.concat([pd.read_csv(f) for f in glob.glob("data/LDAP/*.csv")])
print("LDAP columns:", ldap.columns.tolist())
print("\nUsers per role:")
print(ldap.groupby("role").user_id.nunique().sort_values(ascending=False))

# 2. Insider labels
ins = pd.read_csv("data/insiders.csv")
print("\nInsiders columns:", ins.columns.tolist())
print(ins.head())

dcol = [c for c in ins.columns if "dataset" in c.lower()][0]
ins = ins[ins[dcol].astype(str) == "4.2"]
print("\nr4.2 insiders:", len(ins))

# 3. How many insiders are admins?
admin_roles = [r for r in ldap.role.dropna().unique() if "admin" in r.lower()]
print("\nAdmin-like roles:", admin_roles)
admins = set(ldap[ldap.role.isin(admin_roles)].user_id)
print("Admins:", len(admins))

ucol = [c for c in ins.columns if c.lower() in ("user", "user_id")][0]
both = admins & set(ins[ucol])
print("Insiders who are admins:", len(both), both)
print("\nInsider roles:")
print(ldap[ldap.user_id.isin(ins[ucol])].groupby("role").user_id.nunique())