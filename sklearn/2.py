import os
import tarfile

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    resp = requests.get(housing_url, stream=True)
    f = open(tgz_path, "wb")
    for chunk in resp.iter_content(chunk_size=512):
        if chunk:
            f.write(chunk)
    f.close()
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(house_path=HOUSING_PATH):
    csv_path = os.path.join(house_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    # random list the data
    shuffled_indices = np.random.permulation(len(data))
    test_set_data =  int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_data]
    train_indices = shuffled_indices[test_set_data:]
    return data.iloc[train_indices], data.iloc[test_indices]


def split_test_train(data, test_ratio):
    train_set, test_set = train_test_split(data, test_size=test_ratio, random_state=42)
    return train_set,test_set


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x):
        self.encoder.fit(x)
        return self

    def transform(self, x):
        return self.encoder.transform(x)


housing = load_housing_data()

#  分开训练和测试级
housing["income_cat"] = np.ceil(housing["median_income"] / 5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_train_set = None
strat_test_set = None
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    for set in (strat_test_set, strat_train_set):
        set.drop(["income_cat"], axis=1, inplace=True)

train_housing = strat_train_set.copy()
test_housing = strat_test_set.copy()

corr_matrix = train_housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

train_housing = strat_train_set.drop("median_house_value", axis=1)
train_housing_labels = strat_train_set["median_house_value"].copy()

# 数据填充
median = train_housing["total_bedrooms"].median()
train_housing["total_bedrooms"].fillna(median)

# 数据转换,文本数据转化为数字或者独热向量
imputer = Imputer(strategy="median")
housing_num = train_housing.drop("ocean_proximity", axis=1)
"""
encoder = LabelBinarizer
housing_cat_1hot = encoder.fit_transform(y=housing_num)

# 添加额外的组合属性
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transfrom(housing.values)
"""
# 特征缩放
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ("imputer", Imputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler()),
])
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', MyLabelBinarizer()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])

housing_prepared = full_pipeline.fit_transform(train_housing)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, train_housing_labels)
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(train_housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, train_housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(train_housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

tree_scores = cross_val_score(tree_reg, housing_prepared, train_housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
print(tree_rmse_scores.mean())

line_scores = cross_val_score(lin_reg, housing_prepared, train_housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
line_rmse_scores = np.sqrt(-line_scores)
print(line_rmse_scores.mean())

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, train_housing_labels)
forest_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(train_housing_labels, forest_predictions)
forest_rmse = np.sqrt(forest_mse)


forest_scores = cross_val_score(forest_reg, housing_prepared, train_housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print(forest_rmse_scores.mean())

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, train_housing_labels)

final_model = grid_search.best_estimator_
x_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
x_test_prepared = full_pipeline.transform(x_test)
y_predictions = final_model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, y_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)








