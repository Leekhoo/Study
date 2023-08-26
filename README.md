# Machine Learning 강의 정리
이 레포지토리는 Andrew Ng의 Machine Learning 강의 내용을 정리한 곳입니다. 강의 내용을 요약하고 코드 예제를 포함하여 기록했습니다.

## 강의 목차
1. Machine Learning 소개
   - Machine Learning의 개요와 중요성
   - Supervised Learning, Unsupervised Learning 등 기본 개념 소개

2. Linear Regression
   - Simple Linear Regression와 Multiple Linear Regression
   - Gradient Descent을 이용한 최적화

3. Logistic Regression
   - Binary Classification 문제와 확률 추정
   - Sigmoid Function와 Gradient Descent 적용

4. Overfitting과 Regularization
   - Overfitting 문제의 이해
   - L1, L2 Regularization의 개념과 효과

## 코드 예제
각 주제에 대한 코드 예제는 Octave를 사용하여 작성되었습니다.

### Linear Regression 예제

```octave
% 데이터 로드
data = load('linear_regression_data.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y);

% Linear Regression 모델 학습
X = [ones(m, 1), X];
theta = pinv(X' * X) * X' * y;

% 결과 시각화
plot(X(:, 2), y, 'o');
hold on;
plot(X(:, 2), X * theta);
```

### Logistic Regression 예제

```
% 데이터 로드
data = load('logistic_regression_data.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Logistic Regression 모델 학습
X = [ones(m, 1), X];
initial_theta = zeros(size(X, 2), 1);
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% 결과 시각화
plotData(X(:, 2:3), y);
plotDecisionBoundary(theta, X, y);
```
