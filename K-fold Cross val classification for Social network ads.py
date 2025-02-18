# --------------------- K-Fold Cross Validation ------------------ #

# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset

dataset=pd.read_csv(r"C:\Users\Jan Saida\OneDrive\Documents\Desktop\Excel sheets\Social_Network_Ads.csv")
x=dataset.iloc[:, [2,3]].values
y=dataset.iloc[:,-1].values

# feature scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

# splitting the dataset into training and testing sets

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# training the kernel SVM model on the training set

from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(x_train,y_train)

# predicting the test set results

y_pred=classifier.predict(x_test)

# making the confusion matrix 

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

# accuracy score

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

# bias 

bias = classifier.score(x_train,y_train)
bias


# variance

variance=classifier.score(x_test,y_test)
variance

# applying k-fold cross validation

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=5)
print('Accuracy:{:.2f}%'.format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
