import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats
import joblib
import matplotlib.pyplot as plt

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

housing = pd.read_csv('housing.csv')
housing = housing.head(1000)
housing["income_cat"] = pd.cut(housing['median_income'],
                               bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_idx]
    strat_test_set = housing.loc[test_idx]

for dataset in (strat_train_set, strat_test_set):
    dataset.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

num_attribs = housing.drop("ocean_proximity", axis=1).columns
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)

# Model Evaluation
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores

model_performance = {}
for name, model in models.items():
    model.fit(housing_prepared, housing_labels)
    scores = evaluate_model(model, housing_prepared, housing_labels)
    model_performance[name] = {
        "RMSE Mean": scores.mean(),
        "RMSE Std Dev": scores.std(),
    }

for name, metrics in model_performance.items():
    print(f"{name}:")
    print(f"  - RMSE Mean: {metrics['RMSE Mean']}")
    print(f"  - RMSE Std Dev: {metrics['RMSE Std Dev']}")

# Grid Search for Random Forest
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]
forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=10, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# Feature Importance
feature_importance = grid_search.best_estimator_.feature_importances_
attributes = list(num_attribs) + list(full_pipeline.named_transformers_['cat'].categories_[0])
sorted_importances = sorted(zip(feature_importance, attributes), reverse=True)

plt.figure(figsize=(10, 6))
plt.barh(
    [attr for _, attr in sorted_importances],
    [importance for importance, _ in sorted_importances],
    color="green"
)
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Best Random Forest Model")
plt.gca().invert_yaxis()
plt.show()

# Final Model Evaluation
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
confidence_interval = np.sqrt(
    stats.t.interval(
        confidence,
        len(squared_errors) - 1,
        loc=squared_errors.mean(),
        scale=stats.sem(squared_errors)
    )
)

print(f"Final RMSE: {final_rmse}")
print(f"95% Confidence Interval: {confidence_interval}")

plt.figure(figsize=(10, 7))
plt.scatter(housing["longitude"], housing["latitude"], alpha=0.4,
            c=housing_labels, cmap="viridis", s=housing["population"] / 100, edgecolor="k")
plt.colorbar(label="Median House Value")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographical Distribution of Median House Values")
plt.show()

# Save Best Model
joblib.dump(final_model, "best_housing_model.pkl")
