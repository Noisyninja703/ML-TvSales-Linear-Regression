#Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

#Read our data
data = pd.read_csv("Advertising.csv")
print(data.head())
x = data["TV"].values.reshape(-1, 1)
y = data["Sales"]

#prepare the LR Model
model = LinearRegression()
model.fit(x, y)

#Configure the chart
x_range = np.linspace(x.min(), x.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

#Setup the visualization
fig = px.scatter(data, x='TV', y='Sales', opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Linear Regression'))

#Display the chart
fig.show()