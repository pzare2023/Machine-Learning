
import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

"""
PART 1: basic linear regression
The goal is to predict the profit of a restaurant, based on the number of habitants where the restaurant 
is located. The chain already has several restaurants in different cities. Your goal is to model 
the relationship between the profit and the populations from the cities where they are located.

"""



data = pandas.read_csv('RegressionData.csv', header = None, names=['X', 'y']) 
# Reshape the data so that it can be processed properly
X = data['X'].values.reshape(-1,1) 
y = data['y'] 
# Plot the data using a scatter plot to visualize the data
plt.scatter(X, y) 
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Population vs. Profit')
plt.show()

# Linear regression using least squares optimization
reg = linear_model.LinearRegression() 
reg.fit(X, y) 

# Plot the linear fit
fig = plt.figure()
y_pred = reg.predict(X) 
plt.scatter(X,y, c='b') 
plt.plot(X, y_pred, 'r') 
fig.canvas.draw()
plt.show()


# Predict the profit of a restaurant, if this restaurant is located in a city of 18 habitants 
print("the profit/loss in a city with 18 habitants is ", reg.predict( [[18]] ) [0] )

    

# Load the data from the file 'LogisticRegressionData.csv' in a pandas dataframe. 
data = pandas.read_csv('LogisticRegressionData.csv', header = None, names=['Score1', 'Score2', 'y']) 

# Seperate the data features (score1 and Score2) from the class attribute 
X = data[['Score1', 'Score2']] 
y = data['y'] 

# Plot the data using a scatter plot to visualize the data. 

m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=m[data['y'][i]], color = c[data['y'][i]]) # 2 points
fig.canvas.draw()

# Train a logistic regression classifier to predict the class labels y using the features X
regS = linear_model.LogisticRegression() 
regS.fit(X, y)

# Now, we would like to visualize how well does the trained classifier perform on the training data
# Use the trained classifier on the training data to predict the class labels
y_pred = regS.predict(X) 

m = ['o', 'x']
c = ['red', 'blue'] #this time in red and blue
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=m[y_pred[i]], color = c[y_pred[i]]) # 2 points
fig.canvas.draw()
plt.show()

"""
PART 3: Multi-class classification using logistic regression 

"""

#  One-vs-Rest method (a.k.a. One-vs-All)

reg_multi = LogisticRegression(multi_class='ovr')  

reg_multi.fit(X, y) 

y_pred_multi = reg_multi.predict(X)

print(y_pred_multi)
plt.show()

