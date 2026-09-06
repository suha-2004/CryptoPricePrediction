import pandas as pd
import os

files = [
    "BTCUSDT.csv",
    "ETHUSDT.csv",
    "BNBUSDT.csv",
    "LTCUSDT.csv",
    "XRPUSDT.csv"
]

print("CRYPTOCURRENCY DATA VERIFICATION")
print("=" * 50)

for file in files:
    path = os.path.join("data", file)

    df = pd.read_csv(path)

    print(f"\n{file}")
    print("-" * 30)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Missing values:", df.isnull().sum().sum())
    print("First date:", df["open_time"].iloc[0])
    print("Last date:", df["open_time"].iloc[-1])

print("\n" + "=" * 50)
print("Verification completed!")