# -------------------- Random Forest Classification --------------------- # 



# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset

dataset=pd.read_csv(r"C:\Users\Jan Saida\OneDrive\Documents\Desktop\Excel sheets\Social_Network_Ads.csv")
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

# splitting dataset into training set and testing set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

# training the random forest classification model on the training set

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(max_depth=4,n_estimators=60,random_state=0,criterion='entropy')
classifier.fit(x_train,y_train)

# predicting the test results

y_pred=classifier.predict(x_test)

# making a confusion matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

# accuracy score

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac

# bias 

bias=classifier.score(x_train,y_train)
bias

variance=classifier.score(x_test,y_test)
variance
