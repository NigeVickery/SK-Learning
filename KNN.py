import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#read data and include headers
data = pd.read_csv("C:/Users/nigel/OneDrive/Desktop/Sklearning/car.data", header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

#features
x =  data[['buying',
           'maint',
           'safety',
           'doors',
           'persons',
           'lug_boot'
           ]].values

#labels
y = data[['class']]

#convert data from string

Le = LabelEncoder()
for i in range(len(x[0])):
    x[:, i] = Le.fit_transform(x[:, i])

label_maping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}

y['class'] = y['class'].map(label_maping)
#y = np.array(y)
y= y['class'].values.ravel()


#creating model

knn = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')

#splitting up x and y values for testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

knn.fit(x_train, y_train)
prediction = knn.predict(x_test)

accuracy = metrics.accuracy_score(y_test, prediction)

accuracy_percent = accuracy*100
print("Data:")
print(data.head())
print()
print("Predictions:", prediction)
print()
print("Accuracy: {:.2f}%".format(accuracy_percent))