# Neural Light Transport By using RNN

## 1. Research Vision 

우리의 목표는 Next Generation of Light Transport for Realistic and High-Performance Rendering을 목표로 한다. Light Transport의 반복적이고 재사용 가능한 구조를 발견하고, 물리적 정확성과 높은 계산 효율을 갖춘 새로운 Rendering 방법을 연구한다. 

## 2. Research Problem

Physically Based Rendering은 Rendering Equation을 해결하기 위해 주로 Monte Carlo Path Tracing에 의존한다. 그러나 Monte Carlo Estimation은 제한된 Sample Budget에서 높은 Variance를 발생시키며, 고품질 Rendering을 위해 상당한 계산 비용을 요구한다. 기존 연구들은 Sampling, Caching 및 Information Reuse를 통해 이러한 문제를 완화해 왔다. 그러나 복잡한 Light Transport에 내재된 재사용 가능한 구조를 어떻게 효율적으로 표현하고 실제 계산 비용 감소로 연결할 것인지는 여전히 중요한 연구 문제이다. 본 연구는 Light Transport의 재사용 가능한 구조를 규명하고, 이를 활용하여 정확성과 계산 효율성을 동시에 확보하는 방법을 탐구한다.

## 3. Research Hypothesis

본 연구는 Recurrent Neural Network(RNN)를 활용하여 Multi-Bounce Light Transport의 반복적인 Radiance 계산을 효율적으로 근사하는 방법을 탐구한다. Rendering equation에서 각 표면의 Radiance는 다른 표면으로부터 전달되는 Radiance에 의존하며, 이러한 관계는 반복적인 Light Transport 연산으로 표현된다. 

$$
L = L_e + TL
$$

본 연구는 정적 장면에서 발생하는 이러한 Multi-Bounce Transport 정보를 Offline으로 학습하여, Runtime에서 반복적인 Monte Carlo Integration의 계산 비용을 감소시키고자 한다. 이를 위해 Multi-resolution Hash Grid로 공간적 정보를 표현하고, RNN의 Hidden State를 통해 연속된 Bounce 사이의 Transport 정보를 누적하여 각 지점의 Continuation Radiance를 예측한다. Runtime 에서는 Ray Tracing으로 Geometry와 Visibility를 계산하고, 학습된 RNN으로 Radiance를 근사한다. 이후 Two-Level Monte Carlo를 활용하여 Neural Approximation의 오차를 보정함으로써, 물리적 정확성과 계산 효율성을 동시에 확보하는 것을 목표로 한다.

