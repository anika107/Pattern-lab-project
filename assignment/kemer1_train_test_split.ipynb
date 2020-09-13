import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

x_train_neg = np.load("A.thaliana5289_neg_kemer1.npy")
x_train_pos = np.load("A.thaliana5289_pos_kemer1.npy")
x_test_neg = np.load("A.thaliana1000indep_neg_kemer1.npy")
x_test_pos = np.load("A.thaliana1000indep_pos_kemer1.npy")

y_train_neg = np.tile(0, 5289)
y_train_pos = np.tile(1, 5289)
y_test_neg = np.tile(0, 1000)
y_test_pos = np.tile(1, 1000)

print(x_train_pos.shape)
print(y_train_pos.shape)

seed = 42 # for same size
np.random.seed(seed)

X_train_pos, X_val_pos, Y_train_pos, Y_val_pos = train_test_split(x_train_pos, y_train_pos, test_size=0.20, random_state=seed)
X_train_neg, X_val_neg, Y_train_neg, Y_val_neg = train_test_split(x_train_neg, y_train_neg, test_size=0.20, random_state=seed)

X_train = np.concatenate((X_train_pos, X_train_neg), axis=0)
Y_train = np.concatenate((Y_train_pos, Y_train_neg))

X_val = np.concatenate((X_val_pos, X_val_neg), axis=0)
Y_val = np.concatenate((Y_val_pos, Y_val_neg))

X_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
Y_test = np.concatenate((y_test_pos, y_test_neg))

print(X_test.shape)
print(Y_test.shape)

c = .25
clf = SVC(C=c, random_state=seed) 
clf.fit(X_train, Y_train)

y_train_pred = clf.predict(X_train) 
train_accuracy = accuracy_score(Y_train, y_train_pred)
print(train_accuracy)

y_val_pred = clf.predict(X_val) 
val_accuracy = accuracy_score(Y_val, y_val_pred)
print(val_accuracy)

y_test_pred = clf.predict(X_test) #test set prediction
test_accuracy = accuracy_score(Y_test, y_test_pred)
print(test_accuracy)
