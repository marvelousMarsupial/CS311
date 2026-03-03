import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

CSV = "src/data/daily_csv.csv"
target = "price"

df = pd.read_csv(CSV)
y = df[target].astype(float)

print("Dataset:", CSV)
print("Target:", target)
print("Rows:", len(df))
print("Target stats:\n", y.describe().round(4))

data = pd.DataFrame({f"lag{i}": y.shift(i) for i in range(1, 8)})
data["y"] = y
data = data.dropna()

cut = int(len(data) * 0.8)
train, test = data.iloc[:cut], data.iloc[cut:]

X_train, y_train = train.drop(columns="y"), train["y"]
X_test, y_test = test.drop(columns="y"), test["y"]

m = RandomForestRegressor(n_estimators=200, random_state=0)
m.fit(X_train, y_train)

pred = m.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)

print("Train/Test:", len(train), "/", len(test))
print("MAE:", mae)
print("RMSE:", rmse)

tomorrow = m.predict(data.iloc[[-1]].drop(columns="y"))[0]
print("Tomorrow prediction:", tomorrow)

plt.plot(y_test.index, y_test.values, label="Actual")
plt.plot(y_test.index, pred, label="Predicted")
plt.title("Actual vs Predicted (test split)")
plt.xlabel("Row")
plt.ylabel(target)
plt.legend()
plt.tight_layout()
plt.show()
