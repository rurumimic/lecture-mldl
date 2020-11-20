# Lecture ML/DL

머신러닝 딥러닝 강의 정리

## 과정

- [ML Process](01.ML.Process/README.md)
- [Regression](02.Regression/README.md)
  - 알고리즘: Linear Regression, KNN
  - 회귀모형 평가
- [Classification](03.Classification/README.md)
  - 알고리즘: Logistic Regression, Decision Tree, Artificial Neural Net
- [Hyper Parameter Tuning](04.Hyperparameter/README.md)
  - Random Search
  - Grid Search
- [일반화 성능과 Overfitting](05.Performance/README.md)
  - K-fold Cross Vlidation
  - Overfitting
- [Ensemble](06.Ensemble/README.md)
  - Bagging: Random Forest
  - Boosting: XGBoost
  - Stacking

## 목표

- 모델링을 위한 **전처리** 코드 작성
- **모델링** 코드 작성
- 모델에 대한 **평가**
- 알고리즘의 **원리**와 **모델의 복잡도** 이해
- 모델링의 **성능**을 높이기 위한 방법 이해

---

## 비즈니스 관점에서의 모델 평가

### 평가 전 질문

- 이 모델에서 중요한 것은 무엇인가?
- 무엇을 하려고 했는가?
- 실제 목적에 맞게 모델의 결과를 평가하고 있는가?

### 분류모델 기대가치 평가: 타겟 마케팅

- 타겟 마케팅 사례
  - 고객의 일반적인 응답률 1~2%
  - 프로모션에 응할 확률: P(x)
  - 그 때의 비즈니스 가치: V_1
  - 응하지 않았을 때의 비즈니스 가치: V_0
- 가정
  - 판매는 프로모션을 통해서만 이뤄진다.
- 기대 가치 = P(x) * V_1 + (1 - P(x)) * V_0
  - 상품 판매가: 20,000
  - 매입원가: 10,000
  - 상품 당 프로모션 비용: 200
  - 판매 시 개당 매출이익(공헌이익) = 20,000 - 10,000 - 200 = 9,800
- 기대가치가 0보다 클 것인가?
  - P(x) * 9800 + (1 - P(x)) * (-200) > 0
  - P(x) > 0.02
  - 고객 응답율이 2%보다 높으면 이 프로모션을 진행하는 것이 이익이다
  - 과연 그럴까?

#### 모델 예측 결과

| | 실제 프로모션에 응한 사람 | 실제 응하지 않은 사람 |
|---|---|---|
| 예측 O | 40 | 220 |
| 예측 X | 10 | 730 |

| | 실제 프로모션에 응한 사람 | 실제 응하지 않은 사람 |
|---|---|---|
| 예측 O | 0.04 | 0.22 |
| 예측 X | 0.01 | 0.73 |

#### 비즈니스 가치

| | 실제 프로모션에 응한 사람 | 실제 응하지 않은 사람 |
|---|---|---|
| 예측 O | 9,800 | -200 |
| 예측 X | 0 | 0 |

#### 모델 기대가치 계산

1번(모델 예측 결과) Π 2번(비즈니스 가치)

| | 실제 프로모션에 응한 사람 | 실제 응하지 않은 사람 |
|---|---|---|
| 예측 O | 392 | -44 |
| 예측 X | 0 | 0 |

결과값: 348
