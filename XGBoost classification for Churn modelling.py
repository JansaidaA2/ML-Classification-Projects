# --------------------- XGBoost ------------------------- #

# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset

dataset=pd.read_csv(r"C:\Users\Jan Saida\OneDrive\Documents\Desktop\Excel sheets\Churn_Modelling.csv")
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values
print(x)
print(y)

# encoding catergorical data
# label encoding the "Gender" column

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
print(x)

# one hot encoding the "Geography" column

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
print(x)

# splitting data into training and testing set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# training XGBoost on the training set

from xgboost import XGBClassifier
classifier=XGBClassifier(n_estimator=200,max_depth=4,learning_rate=0.0001)
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

# variance

variance=classifier.score(x_test,y_test)
variance
