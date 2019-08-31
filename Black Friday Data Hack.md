
## Import Library


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
#---------Import dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 550068 entries, 0 to 550067
    Data columns (total 12 columns):
    User_ID                       550068 non-null int64
    Product_ID                    550068 non-null object
    Gender                        550068 non-null object
    Age                           550068 non-null object
    Occupation                    550068 non-null int64
    City_Category                 550068 non-null object
    Stay_In_Current_City_Years    550068 non-null object
    Marital_Status                550068 non-null int64
    Product_Category_1            550068 non-null int64
    Product_Category_2            376430 non-null float64
    Product_Category_3            166821 non-null float64
    Purchase                      550068 non-null int64
    dtypes: float64(2), int64(5), object(5)
    memory usage: 50.4+ MB



```python
 # get number of unique user_id and product id
len(set(train.User_ID)), len(set(train.Product_ID))
```




    (5891, 3631)




```python
#Get number of total user id and product id
len(train.User_ID), len(test.Product_ID)
```




    (550068, 233599)




```python
#check duplicated value
train[train.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Product_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>City_Category</th>
      <th>Stay_In_Current_City_Years</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
#drop dumplicates
train.drop_duplicates()
test.drop_duplicates()
#counting the number of rows after removing the duplicates
train.count()

```




    User_ID                       550068
    Product_ID                    550068
    Gender                        550068
    Age                           550068
    Occupation                    550068
    City_Category                 550068
    Stay_In_Current_City_Years    550068
    Marital_Status                550068
    Product_Category_1            550068
    Product_Category_2            376430
    Product_Category_3            166821
    Purchase                      550068
    dtype: int64




```python
#finding the null values
train.isnull().sum()
```




    User_ID                            0
    Product_ID                         0
    Gender                             0
    Age                                0
    Occupation                         0
    City_Category                      0
    Stay_In_Current_City_Years         0
    Marital_Status                     0
    Product_Category_1                 0
    Product_Category_2            173638
    Product_Category_3            383247
    Purchase                           0
    dtype: int64




```python
#fill na with 0
train.fillna(0,inplace = True)
test.fillna(0,inplace = True)
#check na value
train.isna().sum()
```




    User_ID                       0
    Product_ID                    0
    Gender                        0
    Age                           0
    Occupation                    0
    City_Category                 0
    Stay_In_Current_City_Years    0
    Marital_Status                0
    Product_Category_1            0
    Product_Category_2            0
    Product_Category_3            0
    Purchase                      0
    dtype: int64




```python
train.describe()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-3-39c93376966e> in <module>
    ----> 1 train.describe()
    

    NameError: name 'train' is not defined



```python
#Detecting Outliers
sns.boxplot(train.Occupation)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1938c518>




![png](output_10_1.png)



```python
#Use IQR Score to detect outliers
Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3-Q1
IQR
```




    User_ID               2962.0
    Occupation              12.0
    Marital_Status           1.0
    Product_Category_1       7.0
    Product_Category_2      14.0
    Product_Category_3       8.0
    Purchase              6231.0
    dtype: float64




```python
train = train[~((train < (Q1-1.5*IQR))|(train >(Q3+1.5*IQR))).any(axis = 1)]
train.shape
```




    (543238, 12)



## Exploratory Data Analysis (EDA)
- Part 1: Visulizing purchase distribution with different variables


```python
# Distribution of the target variable
plt.figure(figsize=(12,7))
sns.distplot(train.Purchase, bins = 30)
plt.xlabel("Amount spent in Purchase")
plt.ylabel("Number of Buyers")
plt.title("Purchase amount Distribution")
```




    Text(0.5, 1.0, 'Purchase amount Distribution')




![png](output_14_1.png)



```python
print ("Skew is:", train.Purchase.skew())
print("Kurtosis: %f" % train.Purchase.kurt())
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-0fd0c8792559> in <module>
    ----> 1 print ("Skew is:", train.Purchase.skew())
          2 print("Kurtosis: %f" % train.Purchase.kurt())


    NameError: name 'train' is not defined



```python
#relationship between target and variables
#sns.pairplot(train,hue='Gender',palette='coolwarm')
```


```python
#plot different features against one another\
#Heat map to find the relations between the variables
plt.figure(figsize = (20,10))
c=train.corr()
sns.heatmap(c,cmap ="BrBG", annot = True)
c
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Occupation</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>User_ID</th>
      <td>1.000000</td>
      <td>-0.024173</td>
      <td>0.020490</td>
      <td>0.003624</td>
      <td>0.003611</td>
      <td>0.003815</td>
      <td>0.004271</td>
    </tr>
    <tr>
      <th>Occupation</th>
      <td>-0.024173</td>
      <td>1.000000</td>
      <td>0.024134</td>
      <td>-0.009223</td>
      <td>0.006754</td>
      <td>0.012426</td>
      <td>0.021042</td>
    </tr>
    <tr>
      <th>Marital_Status</th>
      <td>0.020490</td>
      <td>0.024134</td>
      <td>1.000000</td>
      <td>0.020348</td>
      <td>0.000928</td>
      <td>-0.004691</td>
      <td>-0.001296</td>
    </tr>
    <tr>
      <th>Product_Category_1</th>
      <td>0.003624</td>
      <td>-0.009223</td>
      <td>0.020348</td>
      <td>1.000000</td>
      <td>-0.046296</td>
      <td>-0.391783</td>
      <td>-0.341655</td>
    </tr>
    <tr>
      <th>Product_Category_2</th>
      <td>0.003611</td>
      <td>0.006754</td>
      <td>0.000928</td>
      <td>-0.046296</td>
      <td>1.000000</td>
      <td>0.090173</td>
      <td>0.026400</td>
    </tr>
    <tr>
      <th>Product_Category_3</th>
      <td>0.003815</td>
      <td>0.012426</td>
      <td>-0.004691</td>
      <td>-0.391783</td>
      <td>0.090173</td>
      <td>1.000000</td>
      <td>0.290437</td>
    </tr>
    <tr>
      <th>Purchase</th>
      <td>0.004271</td>
      <td>0.021042</td>
      <td>-0.001296</td>
      <td>-0.341655</td>
      <td>0.026400</td>
      <td>0.290437</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_17_1.png)



```python
#Scatter Plot to show relations against features
train_product = sns.scatterplot(x="Product_Category_1", y="Purchase", hue="Gender",data=train)
train_product
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a19e72630>




![png](output_18_1.png)



```python
#Distrition of Age variable
sns.barplot(x="Age",hue = "Purchase", data=train)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-17-e6cdedc93b49> in <module>
          1 #Distrition of Age variable
    ----> 2 sns.barplot(x="Age",hue = "Purchase", data=train)
    

    /anaconda3/lib/python3.6/site-packages/seaborn/categorical.py in barplot(x, y, hue, data, order, hue_order, estimator, ci, n_boot, units, orient, color, palette, saturation, errcolor, errwidth, capsize, dodge, ax, **kwargs)
       3147                           estimator, ci, n_boot, units,
       3148                           orient, color, palette, saturation,
    -> 3149                           errcolor, errwidth, capsize, dodge)
       3150 
       3151     if ax is None:


    /anaconda3/lib/python3.6/site-packages/seaborn/categorical.py in __init__(self, x, y, hue, data, order, hue_order, estimator, ci, n_boot, units, orient, color, palette, saturation, errcolor, errwidth, capsize, dodge)
       1607                                  order, hue_order, units)
       1608         self.establish_colors(color, palette, saturation)
    -> 1609         self.estimate_statistic(estimator, ci, n_boot)
       1610 
       1611         self.dodge = dodge


    /anaconda3/lib/python3.6/site-packages/seaborn/categorical.py in estimate_statistic(self, estimator, ci, n_boot)
       1491                     statistic.append(np.nan)
       1492                 else:
    -> 1493                     statistic.append(estimator(stat_data))
       1494 
       1495                 # Get a confidence interval for this estimate


    /anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py in mean(a, axis, dtype, out, keepdims)
       3116 
       3117     return _methods._mean(a, axis=axis, dtype=dtype,
    -> 3118                           out=out, **kwargs)
       3119 
       3120 


    /anaconda3/lib/python3.6/site-packages/numpy/core/_methods.py in _mean(a, axis, dtype, out, keepdims)
         85             ret = ret.dtype.type(ret / rcount)
         86     else:
    ---> 87         ret = ret / rcount
         88 
         89     return ret


    TypeError: unsupported operand type(s) for /: 'str' and 'int'



```python
#Distribution of Occupation variables
sns.countplot(x="Occupation", data=train)
```


```python
#Distribution of marital status
sns.countplot(x="Marital_Status", data=train)
```


```python
#Distribution of variable Product_Category_1
sns.countplot(x="Product_Category_1", data=train)
```

# Data Pre-Processing


```python
#joining data set
data = pd.concat([train,test], ignore_index = True, sort = False)
data.shape
```


```python
#fill missing value with 0 and check again
data.fillna(0,inplace = True)
data.isnull().sum()
```


```python
#Removing unrelevant columns
data.drop(['User_ID','Product_ID'],axis = 1,inplace = True)
```


```python
#Use apply function to count each variable value
data.apply(lambda x: len(x.unique()))
```


```python
#Frequency Analysis
data.Age.value_counts()
```

# Feature Engineering


```python
# Giving Age Numerical values
age_dict = {"0-17":0, "18-25":1, "26-35":2, "36-45":3, "46-50":4, "51-55":5, "55+":6}
data.Age = data.Age.apply(lambda line: age_dict[line])
data.Age.value_counts()

```


```python
data.Occupation.value_counts()
```


```python
data.City_Category.value_counts()
```


```python
#Converting City_Category to binary
city_dict = {"A":0, "B":1, "C":2}
data.City_Category = data.City_Category.apply(lambda line: city_dict[line])
data.City_Category.value_counts()
```


```python
#Converting Gender to binary
gender_dict = {"F":0, "M":1}
data.Gender = data.Gender.apply(lambda line: gender_dict[line])
data.Gender.value_counts()
```


```python
# Stay_In_Current_City_Years value formatting
data.Stay_In_Current_City_Years.replace("4+", 4,inplace = True)
data.Stay_In_Current_City_Years.value_counts()
data.Stay_In_Current_City_Years = data.Stay_In_Current_City_Years.astype("int64",inplace = True)
```


```python
data.Product_Category_2 = pd.to_numeric(data.Product_Category_2)
data.Product_Category_3 = pd.to_numeric(data.Product_Category_3)
data.Product_Category_2 = data.Product_Category_2.astype("int64",inplace = True)
data.Product_Category_3 = data.Product_Category_3.astype("int64",inplace = True)

```


```python

data.dtypes
```


```python
#build a new dataframe containing only the object columns
#obj_data = data.select_dtypes(include=['object']).copy().astype('int')
#obj_data.head()
```

# Modeling


```python

```


```python

```


```python

```


```python

```

# Train Test Split


```python
from sklearn.model_selection import train_test_split
X = data.drop('Purchase',axis=1)
y = data['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
```

# XGBOOST


```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
#create classifier
classifier = XGBClassifier()
#fit the classifier to the training set
classifier.fit(X_train, y_train)
#Prediction
y_predic_classic = classifier.predict(X_test)
y_predic_classic
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-18-c3e7937cbebb> in <module>
    ----> 1 import xgboost as xgb
          2 from sklearn.metrics import mean_squared_error
          3 from xgboost import XGBClassifier
          4 #create classifier
          5 classifier = XGBClassifier()


    ModuleNotFoundError: No module named 'xgboost'



```python
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predic_classic)

```


```python
#cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator = classifier, X=X_train, y=y_train, cv = 10)
accuracies.mean()
accuracies.std()
```

# Decision Trees


```python
#from sklearn.tree import DecisionTreeClassifier
#dtree = DecisionTreeClassifier()
#dtree.fit(X_train,y_train)
```

## Predictions and Evaluation of Decision Tree
**Create predictions from the test set and create a classification report and a confusion matrix.**


```python
#predictions = dtree.predict(X_test)
```


```python
#from sklearn.metrics import classification_report,confusion_matrix
#print(classification_report(y_test,predictions))
```


```python
#print(confusion_matrix(y_test,predictions))
```

# Random Forest


```python

from sklearn.model_selection import train_test_split
X=data[['Gender', 'Age', 'Occupation', 'City_Category','Stay_In_Current_City_Years','Marital_Status']]  # Features
y=data['Purchase']
#from sklearn.model_selection import train_test_split
#X = data.drop('Purchase',axis=1)
#y = data['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

regressor = RandomForestRegressor(n_estimators=100, random_state = 0)
regressor.fit(X_train, y_train)


```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=100,
                          n_jobs=None, oob_score=False, random_state=0, verbose=0,
                          warm_start=False)




```python

```


```python

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-59-b5a9603961a9> in <module>
          1 from sklearn.metrics import classification_report,confusion_matrix
    ----> 2 print(classification_report(y_test,y_pred))
    

    /anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py in classification_report(y_true, y_pred, labels, target_names, sample_weight, digits, output_dict)
       1850     """
       1851 
    -> 1852     y_type, y_true, y_pred = _check_targets(y_true, y_pred)
       1853 
       1854     labels_given = True


    /anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py in _check_targets(y_true, y_pred)
         79     if len(y_type) > 1:
         80         raise ValueError("Classification metrics can't handle a mix of {0} "
    ---> 81                          "and {1} targets".format(type_true, type_pred))
         82 
         83     # We can't have more than one value on y_type => The set is no more needed


    ValueError: Classification metrics can't handle a mix of multiclass and continuous targets



```python

```


```python

```
