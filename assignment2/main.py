import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

CSV = "src/data/daily_csv.csv"
df = pd.read_csv(CSV)

# target column
target = next((c for c in df.columns if c.strip().lower() in ("close", "price")), None)
if target is None:
    target = df.select_dtypes("number").columns[-1]

y = df[target].astype(float)

# stats
print("Dataset:", CSV)
print("Target:", target)
print("Rows:", len(df))
print("Target stats:\n", y.describe().round(4))

# looks at the last 7 days and guesses the nest
X = pd.concat({f"lag{i}": y.shift(i) for i in range(1, 8)}, axis=1)
data = pd.concat([X, y.rename("y")], axis=1).dropna()

cut = int(len(data) * 0.8)
train, test = data.iloc[:cut], data.iloc[cut:]

m = RandomForestRegressor(n_estimators=200, random_state=0)
m.fit(train.drop(columns="y"), train["y"])

pred = m.predict(test.drop(columns="y"))
mae = mean_absolute_error(test["y"], pred)
rmse = mean_squared_error(test["y"], pred) ** 0.5
print("Train/Test:", len(train), "/", len(test))
print("MAE:", mae)
print("RMSE:", rmse)

tomorrow = m.predict(data.drop(columns="y").tail(1))[0]
print("Tomorrow prediction:", tomorrow)

# chart
plt.figure()
plt.plot(test.index, test["y"].values, label="Actual")
plt.plot(test.index, pred, label="Predicted")
plt.title("Actual vs Predicted (test split)")
plt.xlabel("Row")
plt.ylabel(target)
plt.legend()
plt.tight_layout()
plt.show()
