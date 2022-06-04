import pandas as pd

df = pd.read_csv("train.csv")
df = df[df["year"] == 2008]

l = []
df["temp"] = [f'{y} {m} {d}' for y, m, d in zip(df["year"], df["month"], df["day"])]
for u in df["temp"].unique():
    l.append((u, df[df["temp"] == u]["total"].mean()))

l0 = sorted(l, key=lambda x: x[1], reverse=True)
l1 = sorted(l, key=lambda x: x[1])

l0 = [i[0] for i in l0]
l1 = [i[0] for i in l1]

for i in l0[:15]:
    print(i)

print()

for i in l1[:15]:
    print(i)