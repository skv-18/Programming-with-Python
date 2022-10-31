import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('test.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Dataset being split into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Training set results
viz_train = plt
viz_train.scatter(x_train, y_train, color='blue')
viz_train.plot(x_train, regressor.predict(x_train), color='yellow')
viz_train.title('X VS Ys (Training Set)')
viz_train.xlabel('Y')
viz_train.ylabel('X')
viz_train.show()

# Visualizing the Test set results
viz_test = plt
viz_test.scatter(x_test, y_test, color='blue')
viz_test.plot(x_train, regressor.predict(x_train), color='yellow')
viz_test.title('x VS Ys (Test set)')
viz_test.xlabel('Y')
viz_test.ylabel('X')
viz_test.show()


