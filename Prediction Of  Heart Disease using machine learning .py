#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[4]:


dataset = pd.read_csv('heart.csv')


# In[5]:


dataset.head()


# In[6]:


dataset.info()


# ## Let's check desription of the dataset

# In[7]:


dataset.describe()


# ## Check the features of the dataset

# ## Features of Dataset
# 1. age: age in years
# 2. sex: (1 = male; 0 = female)
# 3. cp: chest pain type (4 values)
# 4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# 5. trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# 6. fbs: fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
# 7. restecg: resting electrocardiographic results (values 0,1,2)
# 8. thalach: maximum heart rate achieved
# 9. exang: exercise induced angina (1 = yes, 0 = no)
# 10. oldpeak: ST depression induced by exercise relative to rest
# 11. slope: the slope of the peak exercise ST segment (0,1,2)
# 12. ca: number of major vessels (0-3) colored by flourosopy
# 13. thal: 1 = normal, 2 = fixed defect, 3 = reversable defect
# 14. target: 0 = no disease, 1 = disease
# 

# In[8]:


dataset.keys()


# In[9]:


dataset.shape


# In[10]:


dataset.isnull().sum()


# In[11]:


dataset.corr()


# ## Understanding the data

# In[12]:


rcParams['figure.figsize'] = 20, 14
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()


# In[13]:


dataset.hist()


# In[14]:


rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')


# In[15]:


x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# In[16]:


x.head()


# In[17]:


y.head()


# In[41]:


dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# In[42]:


standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# In[43]:


y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# ## Decision Tree

# In[44]:


dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))


# In[45]:


plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of maximum features')


# In[40]:


print("The score for Decision Tree Classifier is {}% with {} maximum features.".format(dt_scores[16]*100, [2,4,18]))


# ## Random Forest 

# In[23]:


rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))


# In[24]:


colors = rainbow(np.linspace(0, 1, len(estimators)))
plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)
for i in range(len(estimators)):
    plt.text(i, rf_scores[i], rf_scores[i])
plt.xticks(ticks = [i for i in range(len(estimators))],labels = [str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different number of estimators')


# In[46]:


print("The score for Random Forest Classifier is {}% with {} estimators.".format(rf_scores[1]*100, [100, 500]))


# ## K Neighbors Classifier

# In[47]:


knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))


# In[48]:


plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[50]:


print("The score for K Neighbors Classifier is {}% with {} neighbors.".format(knn_scores[7]*100, 8))


# ## Support Vector

# In[51]:


svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    svc_classifier.fit(X_train, y_train)
    svc_scores.append(svc_classifier.score(X_test, y_test))


# In[52]:


colors = rainbow(np.linspace(0, 1, len(kernels)))
plt.bar(kernels, svc_scores, color = colors)
for i in range(len(kernels)):
    plt.text(i, svc_scores[i], svc_scores[i])
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('Support Vector Classifier scores for different kernels')


# In[53]:


print("The score for Support Vector Classifier is {}% with {} kernel.".format(svc_scores[0]*100, 'linear'))


# We got accuracy for Decision Tree 74% and for Random Forest 84% and for K neighbors Classifier 87% and for Support Vector 83%

# In[ ]:




