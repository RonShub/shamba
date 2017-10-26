
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

training = pd.read_csv('../train.csv') #Called data in youtube
test = pd.read_csv('../test.csv')
train_data_matrix = training.as_matrix()

# print training.head()
#manipulating training data
train_data = train_data_matrix[0:20000, 1:]
train_labels = train_data_matrix[0:20000, 0]

# Creating the Classifier object
classifier = DecisionTreeClassifier()

#training the training data
classifier.fit(train_data, train_labels)

#testing the data
test_data = train_data_matrix[20000:, 1:]
test_labels = train_data_matrix[20000:, 0]

#Plots of specific image
# train_data_matrix = train_data.as_matrix()
d = test_data[8]
d.shape = (28, 28)
pt.imshow(d, cmap='gray')
pt.show()









