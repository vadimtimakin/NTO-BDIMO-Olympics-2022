from prophet import Prophet
import pandas as pd

df = pd.read_csv('train_src.csv')
df["datetime"] = df["datetime"].apply(lambda x: x.replace('.', '-'))
df = df[df["total"] != '?']
df["total"] = df["total"].astype(int)
df = df[df["total"] > 85000]
print(df.head())

df["y"] = df['total']
df["ds"] = df["datetime"]
m = Prophet()
m.fit(df[["ds", "y"]])

future = m.make_future_dataframe(freq='H',periods=4344)
future = future[-4344:]
print(future)

forecasts = m.predict(future)

forecasts = forecasts["yhat"].values
with open('prophet.txt', 'w') as file:
    for i in forecasts.tolist():
        file.write(str(i) + '\n')