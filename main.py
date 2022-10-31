import pandas as pan
import numpy as num
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset train.csv
dataset = pan.read_csv('train.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 4].values
year = dataset["y1"]
print("The mean of Years(before and after scam)", num.mean(x))
print("The mean of Stock Price of Company", num.mean(y))

# Dataset being split into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
plot.scatter(x,y, color="red")
plot.title('Scatter Plot of Stock Price vs Years(before and after scam)')
plot.xlabel('Years(years before mean represents time before renaming the company')
plot.ylabel('Stock Price')
plot.show()

# Representation of Liner Regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Linear Regression Results 
def linear_reg():
    plot.scatter(x, y, color='green')
    plot.plot(x, lin_reg.predict(x), color='yellow')
    plot.title('Relevant or Not Relevant(Linear Regression)')
    plot.xlabel('Scam Level')
    plot.ylabel('Stock Price')
    plot.show()
    return
linear_reg()

# Representation of Polynomial Regression
polynom_reg = PolynomialFeatures(degree=4)
X_poly = polynom_reg.fit_transform(x)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

# Polynomial Regression Results
def polynomial_reg():
    plot.scatter(x, y, color='green')
    plot.plot(x, pol_reg.predict(polynom_reg.fit_transform(x)), color='yellow')
    plot.title('Relevant or Not Relevant(Polinomial Regression)')
    plot.xlabel('Scam Level')
    plot.ylabel('Stock Price')
    plot.show()
    return
polynomial_reg()

# Smoothening the plot line
def polymonial_smooth():
    x_grid = num.arange(min(x), max(x), 0.1)
    x_grid = x_grid.reshape(len(x_grid), 1)
    # Visualizing the Polynomial Regression results
    plot.scatter(x, y, color='green')
    plot.plot(x_grid, pol_reg.predict(polynom_reg.fit_transform(x_grid)), color='yellow')
    plot.title('Relevant or Not Relevant(Polynomial Regression)')
    plot.xlabel('Scam Level')
    plot.ylabel('Stock Price')
    plot.show()
    return

def visualize_difference():
    polynomial_results = pol_reg.predict(polynom_reg.fit_transform(x))
    linear_results = lin_reg.predict(x)
    difference = polynomial_results-linear_results
    plot.plot(difference)
    plot.title('Difference Between Polynomial and Linear Models')
    plot.xlabel('Data point')
    plot.ylabel('Difference')
    plot.show()

polymonial_smooth()
visualize_difference()
print("Predicted Stock Price after 6 Years")
# New Result Prediction after 6 years with Linear Regression
linear_results=lin_reg.predict([[6]])
print("Linear Results",linear_results[0])

# New Result Prediction after 6 years with Polynomial Regression

polynomial_results=pol_reg.predict(polynom_reg.fit_transform([[6]]))
print("Polynomial Results",polynomial_results[0])
