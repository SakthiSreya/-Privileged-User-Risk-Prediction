import pandas as pd, glob

for f in ["logon", "device", "file", "psychometric"]:
    print("\n==", f)
    print(pd.read_csv(f"data/{f}.csv", nrows=3))

print("\n== LDAP files:")
print(glob.glob("data/LDAP/*"))
print(pd.read_csv(glob.glob("data/LDAP/*.csv")[0]).head())