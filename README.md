# Decision Tree Classifier

## Libraries

- Pandas offers data manipulation for numerical tabels

- Matplotlib allows for the display of data in figures and graphs

- Scikit-learn has a few tools to use which are: split training and testing data, tree figures, and the implimentation of the decision tree classifier


```python
import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.tree import plot_tree  

from sklearn.model_selection import train_test_split 
```

## Data Table

Initialize a data frame with the data from the zoo.csv file.


```python
df = pd.read_csv('zoo.csv')

df
```
![output1.jpg](https://github.com/ryankellmer/Project_5160/blob/master/Pictures/output1.jpg)

X will represent the Independent variable. This includes all the attributes from hair to catsize. 


```python
X = df.iloc[:, 1:17]

X.head()
```

![output2.jpg](https://github.com/ryankellmer/Project_5160/blob/master/Pictures/output2.jpg)
Y will represent the Dependent variable. This will be a value between 1 and 7.


```python
y = df.iloc[:, 17]

y.head()
```

## Training and Testing

60% of our data will be used for testing our classifier and 40% to train our classifier.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.6)
```

Initalize a decision tree classifier then proceed to train it with our X _train and y_train data.


```python
clf_df = DecisionTreeClassifier()

clf_df = clf_df.fit(X_train, y_train)
```

Compare our testing data to our trained classifier to see how accuratly it can predict the result.


```python
print("Accuracy Score: ", clf_df.score(X_test, y_test))
```

![output3.jpg](https://github.com/ryankellmer/Project_5160/blob/master/Pictures/output3.jpg)

## Display the Decision Tree Classifier

Print the tree graph with catagorial names.


```python
plt.figure(figsize=(15, 7.5))

plot_tree(clf_df, filled= True, rounded= True, class_names= ["Mammal", "Bird", "Reptile", "Fish", "Amphibian", "Bug", "Invertebrate"], feature_names= X.columns );
```

![output.png](https://github.com/ryankellmer/Project_5160/blob/master/output.png)