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

## 4. Mathematical Formulation and Proposed Method

### 4.1. Recursive Light Transport 

Rendering Equation은 다음과 같이 재귀적 연산자로 표현된다. 

$$
L = L_e + \mathcal{T}L
$$

여기서 $\mathcal{T}$는 Geometry, Visibility 및 material interaction에 의해 결정되는 Light Transport Operator이다. 기존 Path Tracing은 Multi-Bounce Light Transport를 Monte Carlo Sampling으로 반복 추정한다. 본 연구는 이러한 Transport 정보를 Neural Network로 학습하여 반복적인 Radiance Estimation의 계산 비용을 줄이고자 한다. 

### 4.2. Recurrent Neural Light Transport

본 연구는 Multiresolution Hash Grid와 Recurrent Neural Network(RNN)를 결합하여 정적 장면의 Multi-Bounce Radiance를 표현한다. Hash Grid는 공간적 정보를 인코딩 하고, RNN은 연속된 Bounce에서 발생하는 Transport State를 Hidden State에 누적한다.

$$
h_d = \operatorname{RNN}_{\theta}(z_d, h_{d-1})
$$

$$
\widetilde{L}_i = g_{\theta}(z_d, \omega_i, h_d)
$$


여기서 $z_d$는 현재 Bounce의 Position, Direction 및 Material Feature를 포함하며, $h_d$는 이전 Bounce의 정보를 요약한다. 핵심 가설은 제한된 Neural Representation에서 Recurrent Memory가 Radiance Prediction의 정확도와 표현 효율을 개선할 수 있다는 것이다. 

### 4.3. Offline Training

정적 장면에서 Path Tracing을 수행하여 Bounce별 Transport State와 Radiance 데이터를 수집한다. 초기 실험에서는 Continuation Radiance를 학습하며, 최종적으로는 Monte Carlo 보정과 일관된 Incident Radiance Prediction을 목표로 한다.

$$
\mathcal{L}(\theta)
=
\mathbb{E}\left[
\left\|
g_{\theta}(s,H)-\widehat{L}
\right\|_2^2
\right]
$$

학습된 Network Parameters와 Hash Grid는 Runtime에서 재사용한다. RNN의 Hidden State는 각 Light Path를 따라 갱신한다.

### 4.4. Neural-Assisted Monte Carlo Rendering

Runtime에서는 Ray Tracing으로 Geometry와 Visibility를 계산하고, 학습된 RNN을 이용하여 Incident Radiance를 근사한다. Neural Approximation에서 발생하는 오차는 Two-Level Monte Carlo를 이용하여 보정한다.

$$
\widehat{I}
=
\widehat{I}_{\mathrm{Neural}}
+
\widehat{I}_{\mathrm{Residual}}
$$

Neural Estimation은 다수의 저비용 Radiance Query를 수행하며, Residual Estimation은 실제 Path Tracing을 통해 근사 오차를 보정한다. 적절한 Sampling 및 Unbiased Correction 조건 아래에서 물리적으로 정확한 추정기를 구성하는 것을 목표로 한다. 최종적으로 Hash Grid + MLP와 Hash Grid + RNN을 비교하여 Recurrent Memory의 효과를 검증하고, 동일한 Rendering Time에서 Variance Reduction 및 Image Quality 개선 여부를 평가한다.

## 5. Env. Setting 

* Conda environment name: "mi"
* Mitsuba 3.9.1 + drjit 1.5.0 + PyTorch 이용 
* Assets path = "./assets" (cornell box 를 먼저 이용)