import sklearn.linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('test.csv')
df=df.head(200)

# Extract relevant columns
battery_power = df['battery_power']
int_memory = df['int_memory']

# Prepare data for regression
predicting = df[['battery_power', 'int_memory']]
X = np.c_[predicting['battery_power']]
y = np.c_[predicting['int_memory']]

# Plotting
df.plot(kind='scatter', x='battery_power', y='int_memory')
plt.show()

# Train linear regression model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

# Predict using the model
X_new = [[300,100]]  # Ensure this is a 2D list of numeric values
print(model.predict(X_new))
