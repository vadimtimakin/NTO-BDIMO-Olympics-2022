import pandas as pd

df = pd.read_csv('train_src.csv')
test = pd.DataFrame({})
test['datetime'] = [i[:6] + '2009' + i[10:] for i in df["datetime"].iloc[:4344]]
test.to_csv('test_src.csv', index=False)