import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression


X = np.array([
    [3, 10], 
    [4, 5],
    [4, 3],  
    [3, 15], 
    [5, 2],
    [5, 1],
    [4, 2]
])


y = np.array([250, 300, 320, 280, 350, 355, 357])


model = LinearRegression()
model.fit(X, y)

# Coefficients (b1, b2) and intercept (b0)
b0 = model.intercept_
b1, b2 = model.coef_


bedrooms_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 10)
age_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 10)
bedrooms_grid, age_grid = np.meshgrid(bedrooms_range, age_range)


y_pred = b0 + b1 * bedrooms_grid + b2 * age_grid

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


# bedrooms_test = 5
# age_test = 14
# predicted_price = model.predict([[bedrooms_test, age_test]])

# Plot the actual data points
ax.scatter(X[:, 0], X[:, 1], y, color='red', label='Actual Data')

# Plot the regression surface
ax.plot_surface(bedrooms_grid, age_grid, y_pred, alpha=0.5, cmap='viridis')

# # Plot the predicted point for [4 bedrooms, 12 years old]
# ax.scatter(bedrooms_test, age_test, predicted_price, color='blue', s=100, label=f'Predicted: [4 Bedrooms, 12 Years]')


# Set labels
ax.set_xlabel('Number of Bedrooms')
ax.set_ylabel('Age of the House (years)')
ax.set_zlabel('Price')
ax.set_title('Multiple Linear Regression (Bedrooms and Age)')

plt.legend()
plt.show()
