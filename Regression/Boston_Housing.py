# %%

#* Predicting prices of Boston Housing

# NOTE: Using different Regression Models

# Importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing modules for ML
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline # for defining pipelines
from sklearn.preprocessing import StandardScaler # for scaling the data
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
# importing models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# using the grid search for hyperparmeter tuning
from sklearn.model_selection import GridSearchCV

import os

# %%

#* DATA

# Loading data
data_path = os.path.join("datasets","Boston_housing","boston_housing.csv")
housing_data = pd.read_csv(data_path)

# %%
# Exploring the data set

# preview
housing_data.head()
# %%
housing_data.info()
# %%
# checking to see if there is NaN values
housing_data.isnull().values.any()
# %%
# checking the numerical values in data set
housing_data.describe()
# %%
housing_data.hist(bins=50, figsize=(20, 10))
plt.show()

# %%
# Spliting data - exploring the train and test data
train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)

# %%
# examening to see how well the data is divided and represented in both data sets
len(train_set)
len(test_set)

train_set["chas"].value_counts() / len(train_set)
test_set["chas"].value_counts() / len(test_set)

train_set["zn"].value_counts() / len(train_set)
test_set["zn"].value_counts() / len(test_set)

train_set["lstat"].value_counts() / len(train_set)
test_set["lstat"].value_counts() / len(test_set)

# %%
# Creating a new category for better represeation of the data in test and train set
housing_data["lstat_cat"] = pd.cut(housing_data["lstat"], bins=[0., 1.5, 3.0, 4.5, 5., 5.5, 6., 7., np.inf], labels=[1, 2, 3, 4, 5, 6, 7, 8])
housing_data["lstat_cat"].hist()

# %%
split_data = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# %%
for train_index, test_index in split_data.split(housing_data, housing_data["lstat_cat"]):
    strat_train_set = housing_data.loc[train_index] 
    strat_test_set = housing_data.loc[test_index]
    
# %%
# checking the distribution now
strat_test_set["lstat_cat"].value_counts() / len(strat_test_set)
# %%
strat_train_set["lstat_cat"].value_counts() / len(strat_train_set)
# %%
strat_test_set["chas"].value_counts() / len(strat_test_set)
# %%
# Deleting added column
for data_set in (strat_train_set, strat_test_set):
    data_set.drop("lstat_cat", axis=1, inplace=True)
# %%
strat_train_set.head()
# %%
strat_test_set.head()

# %%
#* Discover and visualize the data

housing_training = strat_train_set.copy()
housing_training.plot(kind = "scatter", x = "crim", y = "medv")

# %%
# Looking for corelations in the data
corr_matrix = housing_training.corr()
corr_matrix

# %%
# Since we want to predict housing prices, we are intrested in these corelations
corr_matrix["medv"].sort_values(ascending=False)

# %%
# Ploting a corelations between attributes
scatter_matrix(housing_training, figsize=(20, 20))

# %%
# Ploting just a subset of most influential data
attributes = ["medv", "rm", "zn", "black", "indus", "ptratio", "lstat"]
scatter_matrix(housing_training[attributes], figsize=(20, 20))

# %%
# Ploting just the most influential
housing_training.plot(kind = "scatter", x = "rm", y = "medv")

# %%
housing_training.plot(kind = "scatter", x = "lstat", y = "medv")

# %%
#* Preparing a data for ML training

# deviding data to train and labels
housing_tr = strat_train_set.drop("medv", axis=1)
housing_labels = strat_train_set["medv"].copy()

# %%
# just checking the data
housing_tr.head()
housing_labels.head()

# %%
#* Data cleaning - if needed
# NOTE: Data cleaning is not needed hear, but data scaling is needed

# pipeline for data transformation
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

# transforming the dataset
housing_prepared = num_pipeline.fit_transform(housing_tr)

# %%
housing_prepared

# %%
housing_prepared[0,:]
len(housing_prepared[0,:])

# %%
#* Selecting and Training a Model

#* Linear Regression Model
lin_reg = LinearRegression()

# training a model
lin_reg.fit(housing_prepared, housing_labels)

# %%
# evaluating a model - sample data set

# picking a sample data
sample_train = housing_tr.iloc[:10]
sample_labels = housing_labels.iloc[:10]

# transforming the data
sample_train_pr = num_pipeline.transform(sample_train)

# predictions
print("Predictions: ", lin_reg.predict(sample_train_pr))
print("Labels: ", list(sample_labels))

# %%
# evaluating a model - whole model

# predictions
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("RMSE: ", lin_rmse)

# %%
# evaluating model on test set
housing_test = strat_test_set.drop("medv", axis=1)
housing_test_labels = strat_test_set["medv"].copy()

housing_test_pr = num_pipeline.transform(housing_test)
housing_predictions = lin_reg.predict(housing_test_pr)

lin_mse = mean_squared_error(housing_test_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("RMSE: ", lin_rmse)

# %%
# one more way to evaluate the model
lin_reg.score(housing_test_pr, housing_test_labels)

# %%
#* Saving a model
import joblib

def save_models(main_dir = "models", model_dir="linear_regression", model_name="lin_reg.pkl", model=lin_reg):
    path_model = os.path.join(main_dir,model_dir)
    if not os.path.isdir(path_model):
        os.makedirs(path_model)
    path_model_save = os.path.join(path_model, model_name)
    # saving model
    joblib.dump(model, path_model_save)
    return path_model_save

def load_models(path_model):
    return joblib.load(path_model)

# %%
save_models(main_dir = "models", model_dir="linear_regression_BH", model_name="lin_reg.pkl", model=lin_reg)

# %%
#* Exploring other models

# DecisionTreeRegressor
# model
tree_reg = DecisionTreeRegressor()

# training a model
tree_reg.fit(housing_prepared, housing_labels)

# predictions
housing_predictions = tree_reg.predict(housing_prepared)

# evaluating a model
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("RMSE: ", tree_rmse)

# %%
# evaluating model on test set
housing_predictions = tree_reg.predict(housing_test_pr)
tree_mse = mean_squared_error(housing_test_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("RMSE: ", tree_rmse)

# %%
# Using score to evaluate a model
tree_reg.score(housing_test_pr, housing_test_labels)

# %%
save_models(main_dir="models", model_dir="tree_regression_BH", model_name="tree_reg.pkl", model=tree_reg)

# %%
# Using Cross-Validation - DecisionTreeRegressor

# training and evaluating a model
tree_mse_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

# evaluating RMSE
tree_rmse_scores = np.sqrt(-tree_mse_scores)

# function for printing results
def disply_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
    
# printing results
disply_scores(tree_rmse_scores)

# %%
# Using Cross-Validation - LinearRegression

# training and evaluating a model
lin_mse_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

# evaluating RMSE
lin_rmse_scores = np.sqrt(-lin_mse_scores)

# printing results
disply_scores(lin_rmse_scores)

# %%
# Using Cross-Validation - RandomForestRegressor()

# model
rand_forest = RandomForestRegressor()

# training
rand_forest.fit(housing_prepared, housing_labels)

# model cross validation
rand_forest_scores = cross_val_score(rand_forest, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

# model evaluation - from fitted
housing_pred = rand_forest.predict(housing_prepared)
rand_for_mse = mean_squared_error(housing_labels, housing_pred)
rand_for_rmse = np.sqrt(rand_for_mse)
print("Results for fitted model, RMSE = ", rand_for_rmse)

# model evaluation - from cross validation
rand_fore_score_rmse = np.sqrt(-rand_forest_scores)

# printing results
print("Results form cross validation")
disply_scores(rand_fore_score_rmse)

# %%
# model evaluation - test data set
housing_predictions = rand_forest.predict(housing_test_pr)
rand_forest_mse = mean_squared_error(housing_test_labels, housing_predictions)
rand_forest_rmse = np.sqrt(rand_forest_mse)
print("RMSE: ", rand_forest_rmse)

# %%
rand_forest.score(housing_test_pr, housing_test_labels)

# %%
save_models(main_dir="models", model_dir="rand_fore_regression_BH", model_name="rand_forest.pkl", model=rand_forest)

# %%
#! The best model among investigated is the Random Forest Regression

#* Hyperparameter tuning for Random Forest Regression

param_grid = [
    {'n_estimators': [3, 10, 30, 50, 100], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

# model
rand_forest = RandomForestRegressor()

# defining a grid search
grid_search = GridSearchCV(rand_forest, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

# training a model
grid_search.fit(housing_prepared, housing_labels)

# %%
print(grid_search.best_params_) # best parameters
print(grid_search.best_estimator_) # best estimator
print(grid_search.cv_results_) # seeing all the results

# %%
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# %%
# Final model evaluation
model_best = grid_search.best_estimator_

# prediction on train data
house_pred = model_best.predict(housing_prepared)

# evaluating
model_mse = mean_squared_error(housing_labels, house_pred)
model_rmse = np.sqrt(model_mse)
print("RMSE: ", model_rmse)

# %%
# evaluating model - test data set
housing_predictions = model_best.predict(housing_test_pr)
model_mse = mean_squared_error(housing_test_labels, housing_predictions)
model_rmse = np.sqrt(model_mse)
print("RMSE: ", model_rmse)

# %%
# sample model from test data
sample_test = housing_test.iloc[:10]
sample_test_lables = housing_test_labels.iloc[:10]

# preparing a data
sample_test_pr = num_pipeline.transform(sample_test)

# prediction
sample_pred = model_best.predict(sample_test_pr)

# printing results
print("Prediction: ", sample_pred)
print("Labels: ", list(sample_test_lables))

# %%
#* Analizing features importance

feature_importances = grid_search.best_estimator_.feature_importances_

attributes_list = list(housing_tr)

sorted(zip(feature_importances, attributes_list), reverse=True)

# %%
save_models(main_dir="models", model_dir="tuned_rf_regression", model_name="model_best.pkl", model=model_best)

# %%
model_best.score(housing_test_pr, housing_test_labels)
