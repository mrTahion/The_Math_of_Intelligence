# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradient 

# Importing csv dataset with pandas
data = pd.read_csv('data/GlobalLandTemperaturesByCity.csv')

# Extrating year and month from data
data['year'] = pd.DatetimeIndex(pd.to_datetime(data['dt'])).year
data['month'] = pd.DatetimeIndex(pd.to_datetime(data['dt'])).month


# Using only temp data for Novosibirsk in July(7 month)
data = data .loc[data['City'] == 'Novosibirsk']
data = data.loc[data['month'] == 7]


points = data.as_matrix(['year', 'AverageTemperature'])


def run(lr, num_iter, lam, regularizer=None):
    learning_rate = lr
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = num_iter
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, gradient.compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient.gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations, lam, regularizer)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, gradient.compute_error_for_line_given_points(b, m, points)))
    return b, m

# Predicts y, given x
def predict(b, m, x_values):
    predicted_y = list()
    for x in x_values:
        y = m * x + b
        predicted_y.append(y)
    return predicted_y

# Hyperparameters
learning_rate = 0.0000001
iterations = 500
lamb = 500

# Linear Regression with L1 
b1, m1 = run(learning_rate, iterations, lamb, 'L1')

# Linear Regression with L2 
b2, m2 = run(learning_rate, iterations, lamb, 'L2')

# Linear Regression without Regularization
b3, m3 = run(learning_rate, iterations, lamb)

# plotting
f, ax = plt.subplots(figsize=(14, 5))
ax.set_xlabel('Year')
ax.set_ylabel('Av. Temp in October (Degrees Celcius)')
plt.plot(points[:,0], predict(b1, m1, points[:,0]), label='L1')
plt.plot(points[:,0], predict(b2, m2, points[:,0]), label='L2')
plt.plot(points[:,0], predict(b3, m3, points[:,0]), label='None')
plt.scatter(points[:,0], points[:,1])
plt.legend()
plt.show()

# Making predictions
predictions = predict(b1,m1, [2018, 2055, 2120])

print(predictions)