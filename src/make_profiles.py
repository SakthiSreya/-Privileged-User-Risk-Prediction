import pandas as pd, glob

files = sorted(glob.glob("data/LDAP/*.csv"))
ldap = pd.concat([pd.read_csv(f) for f in files])
ldap = ldap.drop_duplicates("user_id", keep="last")
ldap = ldap.rename(columns={"user_id": "user"})
ldap.to_csv("data/profiles.csv", index=False)
print(ldap.shape)
print(ldap[["user", "employee_name", "role", "department"]].head())