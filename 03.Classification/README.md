# Classification

- Logistic Regression
- Classification: Two Class
  - Logistic Regression
  - 분류 모델 평가
- Classification: Multi Class
  - Decision Tree

## Logisitic Regression

[Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)

- fx = -∞ ~ ∞ -> 0 ~ 1 확률 변환 필요
- odds ratio: 승산. 그 사건이 일어날 가능성 : 일어나지 않을 가능성 비
- log odds: 로그 승산. log(0 ~ ∞) = -∞ ~ ∞

1. 알고리즘의 원리, 개념: 선형판별식 탐색 -> 선형판별식과의 거리를 0~1로 변환
1. 전제조건: NA 처리, feature 정규성, 독립성 가정 충족
1. 문법: sklearn
1. 성능: hyper parameter, 복잡도 결정 요인

### log-odds

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/4a5e86f014eb1f0744e280eb0d68485cb8c0a6c3)

### Logistic function

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/5e648e1dd38ef843d57777cd34c67465bbca694f)

![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/640px-Logistic-curve.svg.png)

---

## Two Class

```py
X.columns.tolist()
model.coef_ # 회귀계수
model.intercept_ # y절편
```

### Confusion Matrix

분류 문제 평가

[Wiki](https://en.wikipedia.org/wiki/Confusion_matrix)

---

## Multi Class

### Decision Tree

[Wiki](https://en.wikipedia.org/wiki/Decision_tree_learning)

- 특정 항목에 대한 의사 결정 규칙 -> 나무 가지 분류
- 직관적
- 계산 비용 낮음
- 화이트 박스 모델

### Gini Index: 지니 계수

[Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)

- 불순도 측정
- 분류 평가
- Information Gain: 정보 증가량. 어떤 속성이 얼마나 많은 정보를 제공하는가

---

## Titanic

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic = pd.read_csv('https://raw.githubusercontent.com/DA4BAM/dataset/master/titanic.0.csv', sep=',', skipinitialspace=True)

# titanic['Survived'].value_counts().plot.bar()
# plt.show()

# plt.hist(titanic['Age'], color = 'green', edgecolor = 'black',bins = 20)
# plt.show()

# plt.hist(titanic['Fare'], color = 'green', edgecolor = 'black',bins = 50)
# plt.show()

# g = sns.FacetGrid(data=titanic, hue='Survived', size = 5)
# g.map(sns.distplot, 'Age', kde=True, hist=False)
# plt.show()

# g = sns.FacetGrid(data=titanic, hue='Survived', size = 5)
# g.map(sns.distplot, 'Fare', kde=True, hist=False)
# plt.show()

# titanic.groupby(['Sex', 'Survived']).size().unstack().plot.bar(stacked=True)
# plt.show()

# df_bar = (titanic.groupby(['Sex','Survived'])['PassengerId'].count()/titanic.groupby(['Sex'])['PassengerId'].count())
# df_bar.unstack().plot.bar(stacked=True)
# plt.show()

# df_bar = (titanic.groupby(['Pclass','Survived'])['PassengerId'].count()/titanic.groupby(['Pclass'])['PassengerId'].count())
# df_bar.unstack().plot.bar(stacked=True)
# plt.show()

# 변수 정리
drop_vars = ['PassengerId','Name','Ticket','Cabin']
titanic0 = titanic.drop(drop_vars, axis=1)
titanic0.head()

# NA 처리
titanic0.isnull().sum()

# Dummy Variable
titanic0.columns

dummy_vars = ['Pclass', 'Sex','Embarked']
for each in dummy_vars:
    dummies = pd.get_dummies(titanic0[each], prefix=each, drop_first=True)
    titanic0 = pd.concat([titanic0, dummies], axis=1)

titanic1 = titanic0.drop(dummy_vars, axis=1)
titanic1.head()

# 데이터 분리
from sklearn.model_selection import train_test_split

X = titanic1.drop('Survived', axis=1)
y = titanic1.iloc[:, 0]

# train+val : test
train_val_x, test_x, train_val_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

# train : val
train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, test_size=0.2, random_state=1)

# Scaling & NA 처리
from sklearn.impute import KNNImputer
imputer = KNNImputer()
imputer.fit(train_x)
train_x = imputer.transform(train_x)
val_x = imputer.transform(val_x)
test_x = imputer.transform(test_x)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

# 모델
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_x, train_y)
val_pred = model.predict(val_x)

# 평가
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

confusion_matrix(val_y, val_pred)
classification_report(val_y, val_pred)
accuracy_score(val_y, val_pred)
```