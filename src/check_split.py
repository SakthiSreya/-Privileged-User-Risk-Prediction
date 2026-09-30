import pandas as pd
df = pd.read_csv("data/features.csv", parse_dates=["day"])
a = df[(df.is_admin == 1) & (df.label == 1)].copy()
a["split"] = pd.cut(a.day, [pd.Timestamp("2000-01-01"), pd.Timestamp("2010-10-01"),
                            pd.Timestamp("2011-01-01"), pd.Timestamp("2030-01-01")],
                    labels=["train", "val", "test"], right=False)
print(a.groupby(["user", "split"], observed=False).size().unstack(fill_value=0))