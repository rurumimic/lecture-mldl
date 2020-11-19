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
  - 부모 Gini - 자식 Gini

1. 가장 정보 증가량이 높은 속성을 첫번째 분할 기준으로 삼는다.
1. 가지를 나누고 과정을 반복한다.

## 통계

- [중심극한정리](https://en.wikipedia.org/wiki/Central_limit_theorem)
- 모분표
- 모수: 특성. 모평균, 모분산. parameter

---

## Artificial Neural Net

- [Wiki](https://en.wikipedia.org/wiki/Artificial_neural_network)
- (신호 * 가중치) 합
- 활성화 Activation: 임계값과 비교
  - Sigmoid, tanh, ReLU, Leaky ReLU, Maxout, ELU
- 편향: bias

### 학습

1. 초기값: 가중치, 편향
1. 예측값과 정답의 오차 계산 -> Loss Function, Cost Function. MSE.
1. 오차의 합 최소화. 가중치 편향 조정
   - 다항식 미분 어려움
   - 경사하강법. 학습률(조정 비율)

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

---

## Iris

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("https://raw.githubusercontent.com/DA4BAM/dataset/master/iris.csv")

iris["Species"].unique() # 'setosa', 'versicolor', 'virginica'
iris['Species'].value_counts() # 50, 50, 50

sns.set_style("whitegrid");
sns.pairplot(iris,hue="Species",size=3);
plt.show()

from sklearn.model_selection import train_test_split

X = iris.drop('Species', axis=1) # iloc[: , :4]
y = iris.iloc[:, -1] # iloc[: , 4]

train_val_x, test_x, train_val_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, test_size=0.2, random_state=1)

# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

# Model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(train_x, train_y)
val_pred = model.predict(val_x)

# Eval
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

confusion_matrix(val_y, val_pred)
# array([[9, 0, 0],
#        [0, 6, 0],
#        [0, 1, 8]])

print(classification_report(val_y, val_pred))
#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00         9
#   versicolor       0.86      1.00      0.92         6
#    virginica       1.00      0.89      0.94         9

#     accuracy                           0.96        24
#    macro avg       0.95      0.96      0.95        24
# weighted avg       0.96      0.96      0.96        24

accuracy_score(val_y, val_pred) 
# 0.95

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(train_x, train_y)

# Visualize
from sklearn import tree
features =['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
Species=['setosa', 'versicolor', 'virginica']

plt.subplots(figsize = (3,3), dpi=400)

tree.plot_tree(model,
               feature_names = features,
               class_names=Species,
               filled = True);

# Visualize
from sklearn.tree import export_graphviz

export_graphviz(model
                , out_file = 'tree.dot'
                , feature_names = X.columns
                , class_names = y.unique()
                , rounded = True, precision = 3, filled = True)
!dot -Tpng tree.dot -o tree.png -Gdpi=300

from IPython.display import Image
Image(filename = 'tree.png', width = 600)

print(classification_report(val_y, val_pred))
#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00         9
#   versicolor       0.86      1.00      0.92         6
#    virginica       1.00      0.89      0.94         9

#     accuracy                           0.96        24
#    macro avg       0.95      0.96      0.95        24
# weighted avg       0.96      0.96      0.96        24
```

---

## MNIST

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

import tensorflow as tf   

data_train, data_test = tf.keras.datasets.mnist.load_data()

(train_x, train_y) = data_train
(test_x, test_y) = data_test

# print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# id = rd.randrange(0,10000)
# print('id = {}'.format(id))
# print('다음 그림은 숫자 {} 입니다.'.format(test_y[id]))
# plt.imshow(test_x[id])
# plt.show()

# np.set_printoptions(linewidth=np.inf)
# print(test_x[id])

# Reshape
print(train_x.shape, test_x.shape) # (60000, 28, 28) (10000, 28, 28)

train_x = train_x.reshape([train_x.shape[0],-1]) # ([.shape[0], 784])
test_x = test_x.reshape([test_x.shape[0],-1])

print(train_x.shape, test_x.shape) # (60000, 784) (10000, 784)

# Scaling
max_num = train_x.max()
train_x = train_x/max_num
test_x = test_x/max_num

# Model
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(train_x, train_y)a
test_pred = model.predict(test_x)
```

```py
# 함수 불러오기
from sklearn.neural_network import MLPClassifier
ANN = MLPClassifier(hidden_layer_sizes=(512,), early_stopping=True ,verbose=True)
ANN.fit(train_x, train_y)
test_pred_ann = ANN.predict(test_x)
accuracy_score(test_y, test_pred_ann)
print(confusion_matrix(test_y, test_pred_ann))
print(classification_report(test_y, test_pred_ann))
```

---

## Fashion MNIST

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

import tensorflow as tf

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

np.set_printoptions(linewidth=np.inf)

labels = ['T-shirt/Top', 'Trouser', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

id = rd.randrange(0,10000)
lid = test_y[id]

print('id = {}'.format(id))
print('다음 그림은 {} 입니다.'.format(labels[lid]) )
plt.imshow(test_x[id])
plt.show()

print(test_x[id])

# Reshape
train_x = train_x.reshape([train_x.shape[0], -1])
test_x = test_x.reshape([test_x.shape[0], -1])

# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

# Neural Net
from sklearn.neural_network import MLPClassifier
ANN = MLPClassifier(hidden_layer_sizes=(512,), early_stopping=True, verbose=True)
ANN.fit(train_x, train_y)
test_pred = ANN.predict(test_x)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(accuracy_score(test_y, test_pred))
print(confusion_matrix(test_y, test_pred))
print(classification_report(test_y, test_pred))
```
