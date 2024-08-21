from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import linear_model

#import the iris data set
iris = datasets.load_iris()

#split into features and labels
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

classes = ['iRis Setosa', 'Iris Versicolour', 'Iris Virginica']

model = svm.SVC()
model.fit(x_train, y_train)

print(model)

predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)

print("Predictions from SVM model: ", predictions)
print("Actual:", y_test)
print("Accuracy from SVM model:", accuracy)
print()
#Create linear regression model
linearreg = linear_model.LinearRegression()

model2 = linearreg.fit(x_train, y_train)
linpredictions = model.predict(x_test)
linaccuracy = accuracy_score(y_test, linpredictions)

print("Predictions from linear regression:", linpredictions)
print("Actual:", y_test)
print("Accuracy from linear regresssion:", linaccuracy)
print("Coefficient of determination:", linearreg.score(x,y))