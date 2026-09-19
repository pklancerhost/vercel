# Code Example 1: Linear Regression & Decision Tree Regressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

X_train = [[1], [2], [3], [4], [5]]
y_train = [2, 4, 6, 8, 10]

# Train Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Train Decision Tree
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)

test_input = [[7]]
lin_pred = lin_model.predict(test_input)
tree_pred = tree_model.predict(test_input)

print("Linear Regression Prediction:", lin_pred[0])
print("Decision Tree Prediction:", tree_pred[0])