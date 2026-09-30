import pandas as pd

ins = pd.read_csv("data/insiders.csv")
print(ins.columns.tolist())
print(ins.head(10))
print("\nRows:", len(ins))