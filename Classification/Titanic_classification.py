#%%
#* Titanic - predict which passangers surrived the Titanic shipwreck

# importing standard modules
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# %%
#* Data

# loading data
data_path = os.path.join("datasets", "Classification", "Titanic_data_set") # folder path
# train data set (with lables)
data_train = os.path.join(data_path, "train.CSV")
# test data set (no lables)
data_test = os.path.join(data_path, "test.CSV")
# lables test
data_test_lables = os.path.join(data_path, "gender_submission.CSV")
# loading data frame
titanic_train = pd.read_csv(data_train)
titanic_test = pd.read_csv(data_test)
titanic_test_labels = pd.read_csv(data_test_lables)

# %%
#* Exploring the data

# exploring the attributes
titanic_train.head()

# %%
# droping the Name column as probably this does not have influence on a surviver outcome
titanic_train_data = titanic_train.drop(["Name"], axis=1)

# %%
titanic_train_data.head()

# %%
# further exploring the data set
titanic_train_data.describe()

# %%
titanic_train_data.info() # there are some values that are NaN, this needs to be fixed

#%%
titanic_train_data["Survived"].value_counts()

# %%
titanic_train_data["Age"].value_counts()

# %%
titanic_train_data["Sex"].value_counts()

# %%
titanic_train_data["Ticket"].value_counts()

# %%
titanic_train_data["Cabin"].value_counts()

# %%
titanic_train_data["Embarked"].value_counts()

# %%
# Droping passangers ID's (think they are not necessary)
titanic_train_data = titanic_train_data.drop(["PassengerId"], axis=1)

# %%
titanic_train_data.hist(bins=50, figsize=(20, 15))
plt.show()

# %%
# looking at correlations between a data
titanic_train_data.corr()

# %%
# droping tickets column
titanic_train_data = titanic_train_data.drop(["Ticket"], axis=1)

# %%
titanic_train_data["Age"].value_counts()

# %%
titanic_train_data.info()

# %%
# checking which columns have NaN values
titanic_train_data["Age"].isnull().values.any() # has NaN values

# %%
titanic_train_data["Cabin"].isnull().values.any() # has NaN values

# %%
# filling the NaN values in age with median values
median = titanic_train_data["Age"].median() # option 3

titanic_train_data["Age"].fillna(median, inplace=True)

# %%
# checking the data set now
titanic_train_data.info()

# %%
# filling the NaN values in Cabin as it is a 'NUL' Cabin
miss_cat = "NUL"
titanic_train_data["Cabin"].fillna(miss_cat, inplace=True)

# %%
# checking the data set now
titanic_train_data.info()

# %%
titanic_train_data["Embarked"].isnull().values.any() # has NaN values

# %%
# filling the NaN values in Embarked as it is a 'N' value
miss_cat = "N"
titanic_train_data["Embarked"].fillna(miss_cat, inplace=True)

# %%
# checking the data set now
titanic_train_data.info()

#%%
# droping the Cabin
titanic_train_data = titanic_train_data.drop(["Cabin"], axis=1)

# %%
titanic_train_data.tail()

# %%
#* Spliting data to train and labels

# training data
titanic_train_dt = titanic_train_data.drop(["Survived"], axis=1)

# training labels
titanic_train_labels = titanic_train_data["Survived"]

# %%
#* Transforming the data

# creating first a numerical train data set
num_attributes = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
titanic_train_num = titanic_train_dt[num_attributes]

# %%
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

#%%
# creating a categorical attributes list
cat_attributes = ["Sex", "Embarked"]

# %%
# full pipeline for transforming the data set
from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', OneHotEncoder(), cat_attributes)
])

# transforming the whole training set
titanic_train_trans = full_pipeline.fit_transform(titanic_train_dt)

# %%
#* Training the model

# Since this is a binary classification trying first - SGDClassifier 
from sklearn.linear_model import SGDClassifier

# model
sgd_clf = SGDClassifier(random_state=42)

# training / fiting
sgd_clf.fit(titanic_train_trans, titanic_train_labels)

# %%
# predicting on one value
sgd_clf.predict([titanic_train_trans[0]])

def print_pred(model, data, value, labels):
    pred = model.predict([data[value]])
    
    if pred[0] == 1:
        print("Predicted: Survived")
    else:
        print("Predicted: Not survived")
    if labels[value] == 1:
        print("Real: Survived")
    else:
        print("Real: Not survived")

print_pred(sgd_clf, titanic_train_trans, 0, titanic_train_labels)

# %%
#* Measuring accuracy of the model

# basic score
sgd_clf.score(titanic_train_trans, titanic_train_labels)

# doing cross valdation
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, titanic_train_trans, titanic_train_labels, cv = 10)

# %%
# Confusion matrix

from sklearn.model_selection import cross_val_predict # importing function that outputs the predictions during cross validation

# runing cross validation predictions
titanic_pred = cross_val_predict(sgd_clf, titanic_train_trans, titanic_train_labels, cv = 10)

# %%
from sklearn.metrics import confusion_matrix # importing function for calculating confusion matrix

confusion_matrix(titanic_train_labels, titanic_pred)

# %%
from sklearn.metrics import precision_score, recall_score

print("Precision score: ", precision_score(titanic_train_labels, titanic_pred))
print("Recall: ", recall_score(titanic_train_labels, titanic_pred))

# %%
from sklearn.metrics import f1_score

print("F1 score: ", f1_score(titanic_train_labels, titanic_pred))

# %%
# Precision vs recall curve

# get the treshold scores 
titanic_socres = cross_val_predict(sgd_clf, titanic_train_trans, titanic_train_labels, cv=10, method="decision_function")

# %%
# compute precision, recall for all the thresholds
from sklearn.metrics import precision_recall_curve

precisions, recall, threshold = precision_recall_curve(titanic_train_labels, titanic_socres)

# %%
# ploting results
def plot_precision_recall_vs_threshold(precisions, recall, threshold):
    plt.plot(threshold, precisions[:-1], "b--", label="Precision")
    plt.plot(threshold, recall[:-1], "g-", label="Recall")
    #[...] # highlight the threshold, add the legend, axis label and grid
    plt.grid(True)
    plt.xlabel('Threshold')
    plt.ylabel('')
    plt.legend()

plot_precision_recall_vs_threshold(precisions, recall, threshold)
plt.show()

# %%
plt.plot(recall, precisions)
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precisions')
plt.show

# %%
from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve(titanic_train_labels, titanic_socres)

# ploting the ROC
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.grid(True)
    plt.xlabel("FPR")
    plt.ylabel("TPR")

plot_roc_curve(fpr, tpr)
plt.show()

# %%
# one way to mesure the accuracy of a classifier is to use the area under the curve (AUC)
from sklearn.metrics import roc_auc_score

roc_auc_score(titanic_train_labels, titanic_socres)

# %%
#* Using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# model
forest_clf = RandomForestClassifier(random_state=42)

# fiting / training
forest_clf.fit(titanic_train_trans, titanic_train_labels)

# %%
#* Precision of the model

titanic_predict = forest_clf.predict(titanic_train_trans)

# %%
# simple score
forest_clf.score(titanic_train_trans, titanic_train_labels)

# %%
# Precision vs Recall

titanic_pred = cross_val_predict(forest_clf, titanic_train_trans, titanic_train_labels, cv = 10)
print("Precision score: ", precision_score(titanic_train_labels, titanic_pred))
print("Recall: ", recall_score(titanic_train_labels, titanic_pred))
print("F1 score: ", f1_score(titanic_train_labels, titanic_pred))

titanic_socres = cross_val_predict(forest_clf, titanic_train_trans, titanic_train_labels, cv=10, method="predict_proba")
titanic_socres = titanic_socres[:,1]
precisions, recall, threshold = precision_recall_curve(titanic_train_labels, titanic_socres)
plot_precision_recall_vs_threshold(precisions, recall, threshold)
plt.show()

#%%
plt.plot(recall, precisions)
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precisions')
plt.show

# %%
fpr, tpr, threshold = roc_curve(titanic_train_labels, titanic_socres)
plot_roc_curve(fpr, tpr)
plt.show()
roc_auc_score(titanic_train_labels, titanic_socres)

# %%
cross_val_score(forest_clf, titanic_train_trans, titanic_train_labels, cv = 10)

# %%
#* Predicting on test data set

titanic_test.head()
# %%
# preparing test data set
dorping_columnts = ["PassengerId", "Name", "Ticket"]

titanic_test_clean = titanic_test.drop(dorping_columnts, axis=1)

# %%
titanic_test_clean.head()

# %%
titanic_test_clean.info()

# %%
# replacing the NaN values
median = titanic_test_clean["Age"].median() # option 3
titanic_test_clean["Age"].fillna(median, inplace=True)

# %%
titanic_test_clean["Fare"].fillna(0, inplace=True)

# %%
titanic_test_clean = titanic_test_clean.drop(["Cabin"], axis=1)

# %%
# checking the test set now
titanic_test_clean.info() # now all valuse are filledin

# %%
# transforming the data set
titanic_test_prepared = full_pipeline.transform(titanic_test_clean)

# %%
titanic_test_pred = forest_clf.predict(titanic_test_prepared)

# %%
titanic_test_labels.head()

# %%
titanic_test_labels = titanic_test_labels.drop(["PassengerId"], axis=1)
