# Hyperparameter Tuning

- 모델링을 최적화하기 위한 파라미터
- Grid Search
  - 명시적인 변수 지정
  - 원하는 경우의 수 모두
- Random Search
  - 범위 안에서 탐색 횟수 지정

## 주의

- 운영환경 성능 보장 X
- 과적합 가능
- 목표: 적절한 예측력 + 적절한 복잡도

---

## Bike

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/bike_sharing.csv'
rides = pd.read_csv(data_path)

# Data prepation
drop_vars = ['instant','dteday','casual','registered']
rides = rides.drop(drop_vars, axis=1)
rides.head()

# Dummy Variable
dummy_fields = ['weekday','season', 'weathersit', 'mnth', 'hr', 'weekday']

for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=1)
    rides = pd.concat([rides, dummies], axis=1)

rides.head()

fields_to_drop = ['season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()

# Data split
test_data = data[-21*24:]
val_data = data[-81*24:-21*24]
train_data = data[:-81*24]

target_fields = 'cnt'
test_x, test_y = test_data.drop(target_fields, axis=1), test_data['cnt']
val_x, val_y = val_data.drop(target_fields, axis=1), val_data['cnt']
train_x, train_y = train_data.drop(target_fields, axis=1), train_data['cnt']

# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

# Modeling KNN
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(train_x, train_y)
knn_pred = model.predict(val_x)

# Eval: rmse, mape
from sklearn.metrics import mean_squared_error, mean_absolute_error
print(mean_squared_error(val_y, knn_pred, squared=False))
print(mean_absolute_error(val_y, knn_pred))

# Comparison
fig, ax = plt.subplots(figsize=(20,5))

ax.plot(test_y, label = 'Real')
ax.plot(knn_pred, label = 'KNN_Prediction')
ax.legend()

dates = pd.to_datetime(rides.iloc[test_data.index]['dteday'])
ax.set_xticks(np.arange(len(dates))[::24])
ax.set_xticklabels(dates[::24], rotation=45)

plt.show()

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(train_x, train_y)
val_pred = model.predict(val_x)
print(mean_absolute_error(val_y, val_pred))
```

---

## Mobile

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mobile = pd.read_csv("https://raw.githubusercontent.com/DA4BAM/dataset/master/mobile_cust_churn.csv")

# dummy
dummy_fields = ['REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN']
for each in dummy_fields:
    dummies = pd.get_dummies(mobile[each], prefix=each, drop_first=True)
    mobile = pd.concat([mobile, dummies], axis=1)

# drop
fields_to_drop = ['id','REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN']
data = mobile.drop(fields_to_drop, axis=1)

# split
from sklearn.model_selection import train_test_split
X = data.drop('CHURN', axis=1)
y = data.iloc[:, 8]
train_val_x, test_x, train_val_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)
train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, test_size=0.2, random_state=1)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

# to np.array
train_y, val_y, test_y = train_y.values, val_y.values, test_y.values

# KNN
# Random Search
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()

from sklearn.model_selection import RandomizedSearchCV
rand_param = { 'n_neighbors' : [3,5,7,9,11,13,15,17,19], 'metric' : ['euclidean', 'manhattan'] }
# rand param: hyperparameter range
# cv = kfold Cross validation
# n iter: try
rand_model = RandomizedSearchCV(knn_model, rand_param, cv=3, scoring='accuracy', n_iter=5)
rand_model.fit(train_x, train_y)
rand_model.cv_results_
rand_model.best_params_
rand_model.best_score_
val_pred = rand_model.predict(val_x)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
accuracy_score(val_y, val_pred)
print(confusion_matrix(val_y, val_pred))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
model = DecisionTreeClassifier()
rand_model = RandomizedSearchCV(model, { 'max_depth': list(range(2,11)), 'min_samples_leaf': list(range(10,51))
)}, cv=3, scoring='accuracy', n_iter=5)
rand_model.fit(train_x, train_y)
rand_model.cv_results_
rand_model.best_params_
rand_model.best_score_
val_pred = rand_model.predict(val_x)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
accuracy_score(val_y, val_pred)

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_param = { 'n_neighbors' : [3,5,7,9,11,13,15,17,19], 'metric' : ['euclidean', 'manhattan'] }
knn = KNeighborsClassifier()
knn_gs = GridSearchCV(knn, grid_param, cv=3, n_jobs=-1) # parallel
knn_gs.fit(train_x, train_y)
knn_gs.best_params_
knn_gs.best_score_
val_pred = knn_gs.predict(val_x)
accuracy_score(val_y, val_pred)
print(confusion_matrix(val_y, val_pred))

# Grid Search 2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

model = DecisionTreeClassifier()
grid_model = GridSearchCV(model, { 'max_depth' : range(2, 7), 'min_samples_leaf': range(10,50,10) }, cv=3, n_jobs=-1)
grid_model.fit(train_x, train_y)
val_pred = grid_model.predict(val_x)
print(accuracy_score(val_y, val_pred))
```
