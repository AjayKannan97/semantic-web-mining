import numpy as np

train_X, train_y = np.load("train50k_dat.npy"), np.load("train50k_labels.npy")
test_X, test_y =  np.load("test_dat.npy"), np.load("test_labels.npy")

from sklearn import svm 
clf = svm.SVC(kernel='linear',verbose=True) # Linear Kernel

#Train the model using the training sets
clf.fit(train_X, train_y)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
