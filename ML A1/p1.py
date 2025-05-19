from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree


### PART 1 ###
# Scikit-Learn provides many popular datasets. The breast cancer wisconsin dataset is one of them. 

X, y = datasets.load_breast_cancer(return_X_y=True) #(4 points) 

# Check how many instances we have in the dataset, and how many features describe these instances
print("There are",X.shape[0], "instances described by", X.shape[1], "features.") #(4 points)  

# Create a training and test set such that the test set has 40% of the instances from the 
# complete breast cancer wisconsin dataset and that the training set has the remaining 60% of  
# the instances from the complete breast cancer wisconsin dataset, using the holdout method. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state = 42)  #(4 points) 

# Create a decision tree classifier. Then Train the classifier using the training dataset created earlier.
# To measure the quality of a split, using the entropy criteria.
clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=6)  #(4 points) 
clf = clf.fit(X_train, y_train)  #(4 points) 

# Apply the decision tree to classify the data 'testData'.
predC = clf.predict(X_test)  #(4 points) 

# Compute the accuracy of the classifier on 'testData'
accuracy= accuracy_score(y_test, predC)
print('The accuracy of the classifier is', accuracy)  #(2 point) 

# Visualize the tree created. Set the font size the 12 (4 points) 
plt.figure(figsize=(20,10)) 
_= plot_tree(clf,filled=True, fontsize=12)  
