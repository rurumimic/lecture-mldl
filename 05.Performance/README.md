## 일반화와 과적합화

- 모든 데이터셋 ⊂ 모집단
- 목표: 부분집합 학습 -> 모집단 적절히 예측
- Generalization: 부분집합(training set) 학습 모델 -> 다른 부분집합에서 적용 가능한가?
- k-fold cross validation: 모든 데이터가 Test 1번, Train k-1번 사용됨
- Learning Curves: 훈련셋의 크기가 커지면 일반화된 성능이 향상된다.
- Fitting Graph: 적합도 그래프로 과적합 측정
  - 모델이 복잡할 수록 과적합화
  - 훈련셋에만 존재하는 특성을 학습하게 된다.

---

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("https://raw.githubusercontent.com/DA4BAM/dataset/master/iris.csv")

# Random Sampling

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Data Split
X = iris.iloc[:, :4]
y = iris.iloc[:, 4]
train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.3)

# Modeling
model = DecisionTreeClassifier(max_depth = 3)
model.fit(train_x, train_y)
model_pred_val = model.predict(val_x)
accuracy_score(val_y, model_pred_val)

# 0.9777
# Loop Avg
result = []
for i in range(1,101):
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.3)

    model = DecisionTreeClassifier(max_depth = 3)
    model.fit(train_x, train_y)
    model_pred_val = model.predict(val_x)
    result.append(accuracy_score(val_y, model_pred_val))

np.mean(result), np.std(result) # (0.9468888888888887, 0.029463515836235476)
np.mean(result) - 2* np.std(result), np.mean(result) + 2* np.std(result) # (0.8879618572164177, 1.0058159205613597)
plt.boxplot(result)
plt.show()

# k-fold Cross Validation

iris = pd.read_csv("https://raw.githubusercontent.com/DA4BAM/dataset/master/iris.csv")
from sklearn.model_selection import train_test_split
X = iris.iloc[:, :4]
y = iris.iloc[:, 4]
train_val_x, test_x, train_val_y, test_y = train_test_split(X, y, test_size=0.2)

# Cross Validation
from sklearn.model_selection import cross_val_score
# Model
model = DecisionTreeClassifier(max_depth = 5)
# All = fit, predict, eval
scores = cross_val_score(model, train_val_x, train_val_y, cv=10)
print(scores)
print(scores.mean()) # 0.9416666666666667
```

---

```py
import numpy as np
import pandas as pd

mobile = pd.read_csv("https://raw.githubusercontent.com/DA4BAM/dataset/master/mobile_cust_churn.csv")

dummy_fields = ['REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN']
for each in dummy_fields:
    dummies = pd.get_dummies(mobile[each], prefix=each, drop_first=True)
    mobile = pd.concat([mobile, dummies], axis=1)

fields_to_drop = ['id','REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN']
data = mobile.drop(fields_to_drop, axis=1)

from sklearn.model_selection import train_test_split
X = data.drop('CHURN', axis=1)
y = data.iloc[:, 8]
train_val_x, test_x, train_val_y, test_y = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_val_x)
train_val_x = scaler.transform(train_val_x)
test_x = scaler.transform(test_x)

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
scores = cross_val_score(model, train_val_x, train_val_y, cv=10)

print(scores)
print(scores.mean()) # 0.611
```

---

## 과적합

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mobile = pd.read_csv("https://raw.githubusercontent.com/DA4BAM/dataset/master/mobile_cust_churn.csv")

dummy_fields = ['REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN']
for each in dummy_fields:
    dummies = pd.get_dummies(mobile[each], prefix=each, drop_first=True)
    mobile = pd.concat([mobile, dummies], axis=1)

    fields_to_drop = ['id','REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN']
data = mobile.drop(fields_to_drop, axis=1)

from sklearn.model_selection import train_test_split

X = data.drop('CHURN', axis=1)
y = data.iloc[:, 8]

train_val_x, test_x, train_val_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)
train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)
train_y, val_y, test_y = train_y.values, val_y.values, test_y.values

# Simple Modeling
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier(max_depth = 2)
model.fit(train_x, train_y)
model_pred_val = model.predict(val_x)
accuracy_score(val_y, model_pred_val)

features = X.columns.values
churn = ['Leave','Stay']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
plot_tree(model,feature_names = features,class_names=churn,filled = True)


######

data_path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/airquality_simple.csv'
air = pd.read_csv(data_path)
air.fillna(method = 'ffill', inplace=True)
from sklearn.model_selection import train_test_split
air_X = air.drop('Ozone', axis=1)
air_y = air.iloc[:, 0]
train_air_x, test_air_x, train_air_y, test_air_y = train_test_split(air_X, air_y, test_size=0.3, random_state=1)
train_air_x, train_air_y = train_air_x.values, train_air_y.values
test_air_x, test_air_y = test_air_x.values, test_air_y.values

# Simple
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
n = 107
model = KNeighborsRegressor(n_neighbors = n)
model.fit(train_air_x, train_air_y)
result = model.predict(train_air_x)

plt.plot(train_air_y) # Real
plt.plot(result, color = 'r') # Predict
plt.show()

# Overfit
n = 1
complex_model = KNeighborsRegressor(n_neighbors = n)
complex_model.fit(train_air_x, train_air_y)
result = complex_model.predict(train_air_x)
plt.plot(train_air_y)
plt.show()
plt.plot(result, color = 'r')
plt.show()

#####

# Fitting Graph
# Decision Tree
result_train = []
result_val = []
for d in range(1,21) :
    model = DecisionTreeClassifier(max_depth = d)
    model.fit(train_x, train_y)
    model_pred_tr,model_pred_val  = model.predict(train_x),model.predict(val_x)
    result_train.append(accuracy_score(train_y, model_pred_tr))
    result_val.append(accuracy_score(val_y, model_pred_val))

pd.DataFrame({'max_depth': list(range(1,21)),'train_acc':result_train, 'val_acc':result_val})

plt.figure(figsize=(15,10))
plt.plot(result_train, label = 'train_acc')
plt.plot(result_val, label = 'val_acc')
plt.xlabel('Complexity')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

########

# KNN Fitting Graph
from sklearn.neighbors import KNeighborsClassifier

result_train = []
result_val = []
for d in range(1,21) :
    model = KNeighborsClassifier(n_neighbors=d)
    model.fit(train_x, train_y)
    model_pred_tr,model_pred_val  = model.predict(train_x),model.predict(val_x)
    result_train.append(accuracy_score(train_y, model_pred_tr))
    result_val.append(accuracy_score(val_y, model_pred_val))

pd.DataFrame({'n_neighbors': list(range(1,21)),'train_acc':result_train, 'val_acc':result_val})

plt.figure(figsize=(15,10))
plt.plot(result_train, label = 'train_acc')
plt.plot(result_val, label = 'val_acc')
plt.xlabel('Complexity')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```