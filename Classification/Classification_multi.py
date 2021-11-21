#%%
#* Multiclass Classification

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
# assing data and labels
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
# Predicting multiple classes using the OvA approach and SGDClassifier

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

# leting the sklearn to decide about the approach for classification
sgd_clf.fit(X_train, y_train)

# predicting
sgd_clf.predict([one_image_digits])

# %%
# printing the scores for all the classes
one_image_scores = sgd_clf.decision_function([one_image_digits])
one_image_scores

# %%
# checking the index with the max score
np.argmax(one_image_scores)

# %%
# printing the classes
sgd_clf.classes_

# %%
sgd_clf.classes_[4]

# %%
# If want to explicitly use OvO or OvA classifiers
from sklearn.multiclass import OneVsOneClassifier

# model
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))

# fiting / training
ovo_clf.fit(X_train, y_train)

# predicting
ovo_clf.predict([one_image_digits])

# %%
len(ovo_clf.estimators_)

#%%
one_image_digits_pred = one_image_digits.reshape(1,-1)

# %%
#* Training a Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

# model
fores_clf = RandomForestClassifier(random_state=42)

# fiting / training
fores_clf.fit(X_train, y_train)

# predict
fores_clf.predict([one_image_digits])

# %%
# this classifier uses probability thus you can print the probability for the given instance
fores_clf.predict_proba([one_image_digits])

# %%
# using cross validation
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# %%
# scaling the inputs by using standrad scaler (like in linear regression models)
from sklearn.preprocessing import StandardScaler

# transforming
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

# %%
# doing cross validation now
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

#%%
#* Analyzing the type of errors that ML model make
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

# %%
from sklearn.metrics import confusion_matrix

# confusion matrix
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

# %%
# ploting the confusion matrix as image
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# %%
# Exploring how well the classifier preforms

# summing how much images there are for each of the class
row_sums = conf_mx.sum(axis=1, keepdims=True)

# %%
norm_conf_mx = conf_mx / row_sums

# %%
# fill the diagonal with zeros and keep only the errors
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

#%%
# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Reshape the array so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
#%%
# Ploting the selected images
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()

# %%
#* Multilabel Classification
# realizes multiple classification categories (not only one)

from sklearn.neighbors import KNeighborsClassifier # supports multilabel classification

# creating multiple classes
y_train_large = (y_train >= 7) # creating a data set with numbers larger then 7
y_train_odd  = (y_train % 2 == 1) # getting only the odd numbers
y_multilabel = np.c_[y_train_large, y_train_odd] # staking the columns

# %%
# model
knn_clf = KNeighborsClassifier()

# fiting / training
knn_clf.fit(X_train, y_multilabel)

# %%
# prediction
knn_clf.predict([one_image_digits])

# %%
from sklearn.metrics import f1_score

# cross validation
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)

# F1 score
f1_score(y_multilabel, y_train_knn_pred, average="macro")

# %%
# assign weight based on number of images of that class availbel in the data set
# F1 score
f1_score(y_multilabel, y_train_knn_pred, average="weighted")

# %%
#* Multioutput Classification

# creating a noise for the images
noise = np.random.randint(0, 100, (len(X_train), 784)) # here 784 as 28 x 28 pixel image is 784
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

# %%
# seeing the image
def plot_image(X_train_mod, image_num):
    one_image_digits = X_train_mod.iloc[image_num]
    one_image_digits = np.array(one_image_digits)
    one_image = one_image_digits.reshape(28, 28)

    plt.imshow(one_image, cmap=mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()
# %%
plot_image(X_test_mod, 3)

# %%
# fiting a model
knn_clf.fit(X_train_mod, y_train_mod)

# predicting
clean_image = knn_clf.predict([X_test_mod.iloc[3]])

# plot digit
def plot_digit(one_image_digits):
    one_image_digits = np.array(one_image_digits)
    one_image = one_image_digits.reshape(28, 28)

    plt.imshow(one_image, cmap=mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()
    
plot_digit(clean_image)

# %%
