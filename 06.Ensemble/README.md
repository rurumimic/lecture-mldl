# Ensemble

## Bagging

**B**ootstrap + **Agg**regating

- Bootstrap: 하나의 데이터셋에서 여러 번 샘플링
  - 같은 훈련 셋을 랜덤 샘플링한 여러 의사결정나무 모델을 평균내서 사용
  - 분류와 회귀 문제를 모두 해결
    - 범주형: Classification
    - 종속변수가 연속형: Regression
- Aggregating: 여러 번 샘플링한 데이터셋으로 모델을 만들고 결과를 집계(평균)

### Random Forest

1. Random Record Selection
   - 전체 = 2/3 **Random** Sampling + 1/3 Validation
1. Random Variable Selection
   - **무작위**로 특정 수의 변수 사용
   - default = √feature 수
1. Out of Bag Error Rate
   - validation 데이터로 검증 = 에러 추정치
1. Vote
   - 분류: 비율 투표
   - 회귀: 평균

- `n_estimators`: 나무 개수
- `bootstrap = True`: 데이터 부트스트래핑 여부
- `max_features = 'auto'`: `default = √feature`

## Boosting

1. 샘플링 -> 모델링 -> 평가
1. 다시 샘플링에 반영 -> 모델링 -> 평가: 오차(노이즈)를 다음 훈련셋으로 사용
1. 반복

1. y = m_1(x) + err
1. y = m_1(x) + err_1(x)
1. y = m_1(x) + m_2(x) + err_2(x)
1. y = m_1(x) + m_2(x) + m_3(x) + err_3(x)

### XGBoost

- learning_rate: 가중치 조절 비율
- max_depth: tree의 depth 제한
- n_estimators: iteration 횟수
- subsample: 학습할 때, 샘플링 비율
- colsample_bytree: tree 만들 때 사용될 feature 비율
- objective: loss function 종류
  - regression: `reg:linear`
  - binary class: `binary:logistic`
  - multi class: `multi:softmax`
  - probability: `multi:softprob`

## Stacking

- 결과를 feature로 사용 -> 모델링
  - Logistic Regression
  - Neural Net
  - SVM
- 새 결과

---

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

iris = pd.read_csv("https://raw.githubusercontent.com/DA4BAM/dataset/master/iris.csv")

from sklearn.model_selection import train_test_split

X = iris.iloc[:, :4]
y = iris.iloc[:, 4]

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10)
rfc.fit(train_x, train_y)
test_pred = rfc.predict(test_x)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(confusion_matrix(test_y, test_pred))
print(classification_report(test_y, test_pred))
```

---

```py
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

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10)
rfc.fit(train_x, train_y)
rfc.oob_score_ # 0.6345
test_pred = rfc.predict(test_x)

print(confusion_matrix(test_y, test_pred))
print(classification_report(test_y, test_pred))

## estimators
acc = []
for n in range(10,200,10) :
    rfc =  RandomForestClassifier()
    rfc.fit(train_x, train_y)
    test_pred = rfc.predict(test_x)
    acc.append(accuracy_score(test_y, test_pred))
    print(n)

print(acc)

## eval chart
plt.figure(figsize=(15,10))
plt.plot(acc)
plt.xlabel('n_tree')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

```py
acc = []
oob = []
for n in range(10,200,10) :
    rfc =  RandomForestClassifier(n_estimators = n, oob_score = True)
    rfc.fit(train_x, train_y)
    test_pred = rfc.predict(test_x)
    acc.append(accuracy_score(test_y, test_pred))
    oob.append(rfc.oob_score_)
    print(n)

plt.figure(figsize=(15,10))
plt.plot(acc)
plt.xlabel('n_tree')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(15,10))
plt.plot(oob)
plt.xlabel('n_tree')
plt.ylabel('OOB Score')
plt.legend()
plt.show()
```

---

## XGBoost

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("https://raw.githubusercontent.com/DA4BAM/dataset/master/iris.csv")

from sklearn.model_selection import train_test_split

X = iris.iloc[:, :4]
y = iris.iloc[:, 4]

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators = 10)
xgb_model.fit(train_x, train_y)
y_pred = xgb_model.predict(test_x)


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(classification_report(test_y, y_pred))
print("-------------------------------------")
print(accuracy_score(test_y, y_pred))
print("-------------------------------------")
print(confusion_matrix(test_y, y_pred))

from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(test_y, y_pred, average='macro')

print(precision, recall)
```

```py
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

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=1)

from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=5)
model.fit(train_x, train_y)
test_pred = model.predict(test_x)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(classification_report(test_y, test_pred))
print(accuracy_score(test_y, test_pred))
print(confusion_matrix(test_y, test_pred))

acc = []
for n in range(5, 200, 5):
  model = XGBClassifier(n_estimators=n)
  model.fit(train_x, train_y)
  test_pred = model.predict(test_x)
  acc.append(accuracy_score(test_y, test_pred))
print(acc)

plt.figure(figsize=(15,10))
plt.plot(acc)
plt.xlabel('n_estimator')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Random search
rand_param ={'max_depth':list(range(2,8)), 'learning_rate':[0.01,0.05,0.1,0.15,0.2], 'n_estimators':list(range(100,601,50)), 'objective':['binary:logistic']}

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

xgb = XGBClassifier()
model = RandomizedSearchCV(xgb, rand_param, cv=3, scoring='accuracy', n_iter=5)
model.fit(train_x, train_y)

print(model.cv_results_)
print(model.best_params_)
print(model.best_score_)

test_pred = model.predict(test_x)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(classification_report(test_y, test_pred))
print(accuracy_score(test_y, test_pred))
print(confusion_matrix(test_y, test_pred))

# 0.686625
# [[2850 1083]
#  [1424 2643]]

# grid search
from sklearn.model_selection import GridSearchCV
grid_param = { 'max_depth' : [3, 5, 8], 'n_estimators': [300, 500, 600], 'objective': ['binary:logistic'], 'learning_rate': [0.01, 0.1, 0.2] }

from xgboost import XGBClassifier
xgb = XGBClassifier()
model = GridSearchCV(xgb, grid_param, cv=3, scoring='accuracy', n_jobs=-1)
model.fit(train_x, train_y)
test_pred = model.predict(test_x)

print(classification_report(test_y, test_pred))
print(accuracy_score(test_y, test_pred))
print(confusion_matrix(test_y, test_pred))
```