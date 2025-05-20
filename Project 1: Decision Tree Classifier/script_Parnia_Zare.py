from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

### PART 1 ###
# Scikit-Learn provides many popular datasets. The breast cancer wisconsin dataset is one of them. 

X, y = datasets.load_breast_cancer(return_X_y=True) 

print("There are",X.shape[0], "instances described by", X.shape[1], "features.")  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state = 42)  

clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=6)  
clf = clf.fit(X_train, y_train) 

# Apply the decision tree to classify the data 'testData'.
predC = clf.predict(X_test)  

# Compute the accuracy of the classifier on 'testData'
accuracy= accuracy_score(y_test, predC)
print('The accuracy of the classifier is', accuracy)  

# Visualize the tree created. Set the font size the 12 
plt.figure(figsize=(20,10)) 
_= plot_tree(clf,filled=True, fontsize=12)  
plt.show()


### PART 2.1 ###
trainAccuracy = []  
testAccuracy = []
# Use the range function to create different depths options, ranging from 1 to 15, for the decision trees
depthOptions = range(1,16) 
for depth in depthOptions: 
    # Use a decision tree classifier that still measures the quality of a split using the entropy criteria.

    cltree = DecisionTreeClassifier(criterion='entropy', max_depth=depth, min_samples_split=6, random_state=42) #(1 point) 

    # Decision tree training
    cltree = cltree.fit(X_train, y_train) 
    # Training error
    y_predTrain = cltree.predict(X_train) 
    # Testing error
    y_predTest = cltree.predict(X_test) 
    # Training accuracy
    trainAccuracy.append(accuracy_score(y_train, y_predTrain)) 
    # Testing accuracy
    testAccuracy.append(accuracy_score(y_test, y_predTest)) 

plt.plot(depthOptions,trainAccuracy,marker='o',color='blue',label='Training Accuracy') 


plt.plot(depthOptions, testAccuracy, marker='x', color='red', label='Test Accuracy')  

plt.legend(['Training Accuracy','Test Accuracy']) 
plt.xlabel('Tree Depth') 
plt.ylabel('Classifier Accuracy') 

plt.show()


### PART 2.2 ###
# Use sklearn's GridSearchCV function to perform an exhaustive search to find the best tree depth and the minimum number of samples to split a node

parameters = {'max_depth':range(1,16),'min_samples_split':range(2, 11)} 

clf = GridSearchCV(DecisionTreeClassifier(criterion='entropy'), parameters,cv=5 ) 
clf.fit(X_train, y_train) 
tree_model = clf.best_estimator_ 
print("The maximum depth of the tree is", clf.best_params_['max_depth'], 
      'and the minimum number of samples required to split a node is', clf.best_params_['min_samples_split'])

# The best model is tree_model. Visualize that decision tree (tree_model). Set the font size the 12 
_ = plot_tree(tree_model,filled=True, fontsize=12) 
plt.show()
