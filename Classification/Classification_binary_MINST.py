#%%
#* Classification - Predicting classes
# NOTE: Predicting handwritten digits by using MINST data set

# importing modules
from numpy.lib.function_base import place
from sklearn.datasets import fetch_openml # for dowloading a MINST data set
import matplotlib as mpl # for ploting the data
import matplotlib.pyplot as plt # for ploting the data
import numpy as np # module for dealing with numeric arrays

# %%
#* Dataset

# dowloading a data set
minst = fetch_openml('mnist_784', version=1)

# exploring the data keys
minst.keys()

# %%
# data and labels
X, y = minst["data"], minst["target"]

print("Data: ", X.shape)
print("Labels: ", y.shape)

# %%
# showing one image form the data set
one_image_digits = X.iloc[0]
one_image_digits = np.array(one_image_digits)
one_image = one_image_digits.reshape(28, 28)

plt.imshow(one_image, cmap=mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

# %%
# checking the label for that image
print("Label:", y[0])

# %%
# converting labels to numbers
y = y.astype(np.uint8)

# %%
#* Spliting the data set into training and testing
#! This should be always done before examening the data!
# NOTE: Here shuffling of the data, and represation of all the needed data in training set is already pre done and prepared in this data set 
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# %%
#* Example of training a binary classifier - only one number
# NOTE: in this example only number 5

# creating a test and training labels contating only True and False
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits
y_test_5 = (y_test == 5)

# %%
#* Training a model - using Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier


# model
sgd_clf = SGDClassifier(random_state=42) # if we want reproductive results we need to set a 'random_state' parameter

# training a model
sgd_clf.fit(X_train, y_train_5)

# predicting a model
one_image_digits_pred = one_image_digits.reshape(1,-1)
sgd_clf.predict(one_image_digits_pred)

# %%
#* Measuring accuracy of the Classification model

# implementing cross-validation function - by writing custom function

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf) # cloning the classification ML model
    # spliting the data into selected folds
    X_train_folds = X_train.iloc[train_index]
    y_train_fold = y_train_5[train_index]
    X_test_folds = X_train.iloc[test_index]
    y_test_folds = y_train_5[test_index]
    
    # training a model
    clone_clf.fit(X_train_folds, y_train_fold)
    
    # predicting
    y_pred = clone_clf.predict(X_test_folds)
    n_correct = sum(y_pred == y_test_folds) # num of correct predictions
    print(n_correct / len(y_pred))

#%%
# cross validation by using the function from sklearn
from sklearn.model_selection import cross_val_score


cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# %%
# writing a simple class to prove that for classification models it is not good to use accuracy as a measure

from sklearn.base import BaseEstimator


class Never5Classifier(BaseEstimator):
    def fit(self, X, y = None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

    
never_5_clf = Never5Classifier()

cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# %%
#* Confusion Matrix

from sklearn.model_selection import cross_val_predict # importing function that outputs the predictions during cross validation


# runing cross validation predictions
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3)

# %%
from sklearn.metrics import confusion_matrix # importing function for calculating confusion matrix


confusion_matrix(y_train_5, y_train_pred)

# %%
#* Precision and Recall
from sklearn.metrics import precision_score, recall_score


print("Precision score: ", precision_score(y_train_5, y_train_pred))
print("Recall: ", recall_score(y_train_5, y_train_pred))

# %%
# both precision score and recall are combined in F1 score
from sklearn.metrics import f1_score


print("F1 score: ", f1_score(y_train_5, y_train_pred))

# %%
#* Changing the treshold scores for selected classifier

y_scores = sgd_clf.decision_function([one_image_digits])
print(y_scores)

threshold = 0
y_some_digits_pred = (y_scores > threshold)
print(y_some_digits_pred)

# %%
# raising the threshold
threshold = 8000
y_some_digits_pred = (y_scores > threshold)
print(y_some_digits_pred)

# %%
#* Which treshold to use?

# get the treshold scores 
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

# %%
# compute precision, recall for all the thresholds
from sklearn.metrics import precision_recall_curve


precisions, recall, threshold = precision_recall_curve(y_train_5, y_scores)

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
# precision vs recall
plt.plot(recall, precisions)
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precisions')
plt.show

# %%
# search for a threshold that gives at least 90% precision
threshold_90_precision = threshold[np.argmax(precisions >= 0.90)]

# %%
# to predict and evalute your training model now, no need to use predict()
y_train_pred_90 = (y_scores >= threshold_90_precision)

# %%
# checking the precision and recall
precision_score(y_train_5, y_train_pred_90)

recall_score(y_train_5, y_train_pred_90)

# %%
# ploting the ROC curve
from sklearn.metrics import roc_curve


fpr, tpr, threshold = roc_curve(y_train_5, y_scores)

# %%
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

roc_auc_score(y_train_5, y_scores)

# %%
#* Using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier


forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

# %%
# need to use the scores for the ROC curve
y_scores_forest = y_probas_forest[:,1]

fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

# %%
# AUC surface below the curve
roc_auc_score(y_train_5, y_scores_forest)

# %%
y_train_pred_f = y_scores_forest>=0.5
precision_score(y_train_5, y_train_pred_f)

# %%
recall_score(y_train_5, y_train_pred_f)
