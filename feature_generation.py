import pandas as pd
import datetime
import numpy as np
from tqdm import tqdm

weekdays = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6,
}

MAIN = pd.read_csv('train_src.csv')
MAIN = MAIN[MAIN["total"] != '?'] # Dealing with missing targets

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Time 
    df["time"] = [int(i.split()[1][:2]) for i in df["datetime"].values]

    # Month
    df["month"] = [int(i.split()[0].split('.')[1]) for i in df["datetime"].values]

    # Day
    df["day"] = [int(i.split()[0].split('.')[0]) for i in df["datetime"].values]

    # Year
    df["year"] = [int(i.split()[0].split('.')[2]) for i in df["datetime"].values]

    # Weekday
    df["weekday"] = [datetime.date(y, m, d).strftime("%A") for y, m, d in zip(df["year"], df["month"], df["day"])]
    df["weekday"] = df["weekday"].map(weekdays)

    # Season
    season = []
    for i in df["month"].values:
        if i in [6, 7, 8]:
            season.append(0)
        elif i in [9, 10, 11]:
            season.append(1)
        elif i in [12, 1, 2]:
            season.append(2)
        elif i in [3, 4, 5]:
            season.append(3)
    df["season"] = season

    # Weekend
    df["is_weekend"] = [int(i in [5, 6]) for i in df["weekday"].values]

    # This hour this day in last year
    l = []
    prev = None
    for idx in tqdm(range(len(df))):
        i = df.iloc[idx]
        if i["year"] == 2005:
            l.append(int(i["total"]) / 0.942336739)
        elif i["year"] == 2009:
            slc = MAIN[MAIN['datetime'] == i['datetime'].replace(str(i["year"]), '2008')]["total"].values
            if len(slc) == 1:
                l.append(slc[0])
                prev = slc[0]
            else:
                l.append(prev)
        else:
            slc = df[df['datetime'] == i['datetime'].replace(str(i["year"]), str(int(i["year"])-1))]["total"].values
            if len(slc) == 1:
                l.append(slc[0])
            else:
                l.append(int(i["total"]) / 0.942336739)
    df["this_hour_last_year"] = l

    return df

df = pd.read_csv("train_src.csv")
df = df[df["total"] != '?'] # Dealing with missing targets
df = process_dataframe(df)
df.to_csv('train.csv')
print(df.head())

test = pd.read_csv('test_src.csv')
test = process_dataframe(test)
test.to_csv('test.csv')
print(test.head())