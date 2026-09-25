# Neural Light Transport

## Foundations and Operator Theory

##### Goral et al. 1984. Modeling the Interaction of Light Between Diffuse Surfaces

- [DOI: 10.1145/800031.808601](https://doi.org/10.1145/800031.808601)
- Diffuse surface 사이의 interreflection을 form factor 기반으로 계산하는 초기 Radiosity 연구이다.
- 계산된 surface illumination이 viewpoint와 독립적이어서 여러 view에서 재사용될 수 있다.
- Scene geometry에 의해 결정되는 transport interaction을 명시적으로 계산한 초기 연구이다.


##### Kajiya 1986. The Rendering Equation

- [DOI: 10.1145/15886.15902](https://doi.org/10.1145/15886.15902)
- 빛의 emission, reflection, transport를 하나의 Rendering Equation으로 통합하였다.
- Monte Carlo integration을 이용한 Path Tracing을 제시하였다.
- 이후 대부분의 physically based light transport 연구의 이론적 출발점이다.


##### Soler et al. 2022. A Theoretical Analysis of Compactness of the Light Transport Operator

- [DOI: 10.1145/3528233.3530725](https://doi.org/10.1145/3528233.3530725)
- 일반적인 Light Transport Operator의 compactness를 함수해석학적으로 분석하였다.
- 일반 LTO는 non-compact할 수 있으므로 고정된 finite-dimensional representation으로 항상 잘 근사할 수 있다고 보장할 수 없다.
- Global transport compression과 local 또는 restricted approximation을 구분해야 하는 이론적 근거를 제공한다.


##### Soler and Subr 2025. Spectral Theory of Light Transport Operators

- [DOI: 10.1145/3774756](https://doi.org/10.1145/3774756)
- Light Transport Operator의 spectrum과 compact approximation을 분석한다.
- Operator spectrum과 반복적인 circular light paths 사이의 관계를 연구한다.
- Spectrum이 존재한다는 사실과 소수의 mode로 전체 transport를 표현할 수 있다는 주장은 구분해야 한다.


## Radiosity, Hierarchical Transport, and Precomputation

##### Ward et al. 1988. A Ray Tracing Solution for Diffuse Interreflection

- [DOI: 10.1145/378456.378490](https://doi.org/10.1145/378456.378490)
- Monte Carlo로 계산한 diffuse indirect illumination을 공간적으로 저장하고 주변 위치에서 재사용한다.
- 모든 surface point에서 hemisphere integration을 반복하지 않아도 되도록 한다.
- 이후 Irradiance Caching 계열에서 사용되는 spatial coherence의 기초가 되었다.


##### Hanrahan et al. 1991. A Rapid Hierarchical Radiosity Algorithm

- [DOI: 10.1145/122718.122740](https://doi.org/10.1145/122718.122740)
- Surface 사이의 form-factor interaction을 계층적인 구조로 계산한다.
- 모든 patch pair를 동일한 해상도로 계산하지 않고 필요한 interaction만 세분화한다.
- Light transport interaction의 중요도와 scale에 따라 계산 정밀도를 adaptive하게 조절한다.


##### Gortler et al. 1993. Wavelet Radiosity

- [DOI: 10.1145/166117.166146](https://doi.org/10.1145/166117.166146)
- Radiosity transport를 wavelet basis로 표현한다.
- Smooth한 transport 영역에서 작은 wavelet coefficients를 제거해 sparse representation을 만든다.
- Visibility discontinuity와 같은 high-frequency structure에서는 더 많은 coefficients가 필요하다.


##### Jensen 1996. Global Illumination Using Photon Maps

- [DOI: 10.1007/978-3-7091-7484-5_3](https://doi.org/10.1007/978-3-7091-7484-5_3)
- Light source에서 생성한 photon을 scene에 저장하고 camera-side query에서 재사용한다.
- Caustics와 diffuse indirect illumination을 density estimation으로 계산한다.
- Camera와 독립적인 light-subpath information을 scene space에 저장해 재사용한다.


##### Keller 1997. Instant Radiosity

- [DOI: 10.1145/258734.258769](https://doi.org/10.1145/258734.258769)
- Indirect illumination을 Virtual Point Lights의 집합으로 근사한다.
- 복잡한 multi-bounce transport를 여러 개의 direct-lighting 문제로 바꾼다.
- 이후 many-light methods와 Lightcuts 계열의 기반이 되었다.


##### Sloan et al. 2002. Precomputed Radiance Transfer for Real-Time Rendering in Dynamic, Low-Frequency Lighting Environments

- [DOI: 10.1145/566570.566612](https://doi.org/10.1145/566570.566612)
- Static scene의 light transport를 Spherical Harmonics transfer coefficients로 사전 계산한다.
- Runtime에는 lighting coefficients와 precomputed transfer의 내적으로 shading을 계산한다.
- Low-frequency illumination이라는 제한 조건 아래 scene transport를 강하게 재사용한다.


##### Sloan et al. 2003. Clustered Principal Components for Precomputed Radiance Transfer

- [DOI: 10.1145/882262.882281](https://doi.org/10.1145/882262.882281)
- PRT transport vectors를 spatial cluster별 PCA basis로 압축한다.
- 각 surface point의 고차원 transport를 소수의 local basis coefficients로 표현한다.
- Global basis 하나보다 local low-dimensional structure가 더 효율적일 수 있음을 보여준다.


##### Walter et al. 2005. Lightcuts: A Scalable Approach to Illumination

- [DOI: 10.1145/1073204.1073318](https://doi.org/10.1145/1073204.1073318)
- 많은 Virtual Point Lights를 hierarchical light tree로 구성한다.
- Error bound를 이용해 여러 light를 하나의 cluster로 근사할 수 있는 영역을 결정한다.
- 모든 light contribution을 직접 계산하지 않고 필요한 interaction만 adaptive하게 평가한다.


##### Walter et al. 2006. Multidimensional Lightcuts

- [DOI: 10.1145/1141911.1141997](https://doi.org/10.1145/1141911.1141997)
- Lightcuts의 clustering 개념을 gather points와 추가적인 rendering dimensions로 확장한다.
- Motion blur, depth of field, participating media와 같은 고차원 적분에도 hierarchical approximation을 적용한다.
- High-dimensional rendering integral에서도 adaptive clustering이 가능함을 보여준다.


##### Hašan et al. 2007. Matrix Row-Column Sampling for the Many-Light Problem

- [DOI: 10.1145/1276377.1276410](https://doi.org/10.1145/1276377.1276410)
- Surface samples와 많은 light 사이의 contribution을 하나의 matrix로 표현한다.
- 전체 matrix를 계산하지 않고 일부 rows와 columns만 sampling하여 중요한 구조를 추정한다.
- Light transport matrix에 존재하는 redundancy를 실제 계산 감소에 활용한다.


##### Mahajan et al. 2007. A Theory of Locally Low Dimensional Light Transport

- [DOI: 10.1145/1276377.1276454](https://doi.org/10.1145/1276377.1276454)
- Local image 또는 surface patch에서 light transport의 dimensionality가 어떻게 변하는지 분석한다.
- Glossy reflection, shadow와 spatial patch size가 필요한 representation dimension에 미치는 영향을 연구한다.
- Physical condition과 local transport dimensionality를 직접 연결한다는 점에서 중요한 연구이다.


##### Lehtinen et al. 2008. A Meshless Hierarchical Representation for Light Transport

- [DOI: 10.1145/1360612.1360636](https://doi.org/10.1145/1360612.1360636)
- Mesh connectivity에 의존하지 않는 hierarchical illumination representation을 제안한다.
- Illumination complexity에 맞추어 spatial resolution을 adaptive하게 조절한다.
- Scene 전체에 동일한 해상도의 transport representation을 사용하는 것을 피한다.


##### Huang and Ramamoorthi 2010. Sparsely Precomputing the Light Transport Matrix for Real-Time Rendering

- [DOI: 10.1111/j.1467-8659.2010.01729.x](https://doi.org/10.1111/j.1467-8659.2010.01729.x)
- 일부 surface vertices의 transport만 충분히 계산하고 나머지는 sparse하게 sampling한다.
- Nearby transport의 locally low-rank structure를 이용해 계산하지 않은 matrix elements를 복원한다.
- Transport를 모두 계산한 뒤 압축하는 대신 처음부터 필요한 transport 계산 자체를 줄인다.


##### Lessig and Fiume 2010. On the Effective Dimension of Light Transport

- [DOI: 10.1111/j.1467-8659.2010.01736.x](https://doi.org/10.1111/j.1467-8659.2010.01736.x)
- Light transport를 표현하는 데 필요한 effective dimension을 이론적으로 분석한다.
- Frequency content와 representation dimensionality 사이의 관계를 연구한다.
- Light transport의 computational 또는 representational simplicity를 정량화하려는 연구와 직접적으로 관련된다.


## Frequency, Smoothness, and Local Complexity

##### Křivánek et al. 2005. Radiance Caching for Efficient Global Illumination Computation

- [DOI: 10.1109/TVCG.2005.83](https://doi.org/10.1109/TVCG.2005.83)
- Diffuse뿐 아니라 smooth glossy surface의 incoming radiance를 공간과 방향에 대해 cache한다.
- Nearby query 사이의 radiance coherence를 이용해 repeated integration을 줄인다.
- BRDF 특성과 local variation에 따라 cache interpolation 가능 범위를 결정한다.


##### Durand et al. 2005. A Frequency Analysis of Light Transport

- [DOI: 10.1145/1073204.1073320](https://doi.org/10.1145/1073204.1073320)
- Light transport를 space-angle Fourier domain에서 분석한다.
- Propagation, reflection, occlusion이 light-field frequency를 각각 어떻게 변화시키는지 설명한다.
- Geometry, visibility, BRDF와 transport bandwidth 사이의 관계를 수학적으로 분석한다.


##### Schwarzhaupt et al. 2012. Practical Hessian-Based Error Control for Irradiance Caching

- [DOI: 10.1145/2366145.2366212](https://doi.org/10.1145/2366145.2366212)
- Irradiance의 second-order derivative인 Hessian을 이용해 interpolation error를 추정한다.
- Local illumination이 빠르게 변하는 곳에서는 cache reuse 범위를 줄인다.
- Local variation을 실제 reuse와 recomputation의 결정 기준으로 사용한다.


##### Belcour et al. 2013. 5D Covariance Tracing for Efficient Defocus and Motion Blur

- [DOI: 10.1145/2487228.2487239](https://doi.org/10.1145/2487228.2487239)
- Local light-field spectrum을 covariance matrix로 근사한다.
- Light path의 optical events를 따라 bandwidth와 anisotropy가 어떻게 변하는지 추적한다.
- 추정된 local signal complexity를 sampling과 reconstruction에 활용한다.


##### Belcour et al. 2014. A Local Frequency Analysis of Light Scattering and Absorption

- [DOI: 10.1145/2629490](https://doi.org/10.1145/2629490)
- Participating media에서 scattering과 absorption이 local light-field frequency를 어떻게 변화시키는지 분석한다.
- Spectrum covariance를 light path를 따라 propagate하는 analytical model을 제시한다.
- Physical interaction과 local transport frequency complexity 사이의 관계를 연구한다.


##### Wang et al. 2014. Parallel and Adaptive Visibility Sampling for Rendering Dynamic Scenes with Spatially Varying Reflectance

- [DOI: 10.1016/j.cag.2013.10.036](https://doi.org/10.1016/j.cag.2013.10.036)
- Dynamic scene에서 visibility를 spatial/angular domain에 따라 adaptive하게 sampling한다.
- Visibility가 단순한 영역과 복잡한 영역에 서로 다른 수의 rays를 할당한다.
- Visibility complexity를 직접 computation allocation에 활용한다.


## Path Guiding and Statistical Structure

##### Vorba et al. 2014. On-line Learning of Parametric Mixture Models for Light Transport Simulation

- [DOI: 10.1145/2601097.2601203](https://doi.org/10.1145/2601097.2601203)
- Rendering 중 수집한 samples로 incident radiance distribution을 online 학습한다.
- Parametric mixture model을 이용해 contribution이 높은 방향을 더 자주 sampling한다.
- Radiance 자체보다 reusable sampling distribution을 학습한다.


##### Müller et al. 2017. Practical Path Guiding for Efficient Light-Transport Simulation

- [DOI: 10.1111/cgf.13227](https://doi.org/10.1111/cgf.13227)
- Scene의 spatio-directional incident-radiance distribution을 SD-tree로 학습한다.
- 학습된 distribution을 sampling PDF로 사용해 low-contribution paths를 줄인다.
- Spatial과 directional statistical structure를 Monte Carlo variance reduction에 이용한다.


##### Dahm and Keller 2017. Learning Light Transport the Reinforced Way

- [DOI: 10.1145/3084363.3085032](https://doi.org/10.1145/3084363.3085032)
- Reinforcement learning과 light transport simulation을 연결한다.
- Rendering 과정에서 좋은 sampling decisions를 학습하여 높은 contribution의 paths를 더 자주 생성한다.
- Learned representation을 radiance prediction보다 transport sampling 개선에 사용한다.


##### Herholz et al. 2019. Volume Path Guiding Based on Zero-Variance Random Walk Theory

- [DOI: 10.1145/3230635](https://doi.org/10.1145/3230635)
- Participating media에서 zero-variance random walk의 조건을 분석한다.
- Direction, distance, termination 등 여러 path-sampling decisions를 함께 guiding한다.
- Path Guiding을 단순 directional sampling이 아니라 전체 random-walk optimization 문제로 확장한다.


## Specular and Path-Space Structure

##### Jakob and Marschner 2012. Manifold Exploration: A Markov Chain Monte Carlo Technique for Rendering Scenes with Difficult Specular Transport

- [DOI: 10.1145/2185520.2185554](https://doi.org/10.1145/2185520.2185554)
- Valid specular paths가 path space에서 geometric constraint manifold를 형성한다는 점을 이용한다.
- Blind random sampling 대신 valid path 주변을 manifold perturbation으로 탐색한다.
- Spatially high-frequency한 transport도 path-space에서는 structured할 수 있음을 보여준다.


##### Hanika et al. 2015. Manifold Next Event Estimation

- [DOI: 10.1111/cgf.12681](https://doi.org/10.1111/cgf.12681)
- Refractive 또는 reflective specular chain을 통과해 light source와 연결되는 경로를 계산한다.
- 일반적인 Next Event Estimation이 찾기 어려운 specular transport를 manifold solver로 탐색한다.
- Specular path의 geometric constraint를 Monte Carlo estimator와 결합한다.


##### Zeltner et al. 2020. Specular Manifold Sampling for Rendering High-Frequency Caustics and Glints

- [DOI: 10.1145/3386569.3392408](https://doi.org/10.1145/3386569.3392408)
- 두 endpoints 사이를 연결하는 specular subpath를 stochastic manifold sampling으로 생성한다.
- Caustics, multiple refraction, glints와 같은 어려운 specular transport를 처리한다.
- Spatial smoothness가 없는 transport에서도 path-space structure를 활용할 수 있음을 보여준다.


##### Wang et al. 2020. Path Cuts: Efficient Rendering of Pure Specular Light Transport

- [DOI: 10.1145/3414685.3417792](https://doi.org/10.1145/3414685.3417792)
- Pure-specular multi-bounce transport를 path family 단위로 다룬다.
- 많은 specular paths를 path cuts라는 구조를 이용해 효율적으로 표현한다.
- Difficult specular transport를 개별 random paths가 아니라 structured path set으로 처리한다.


## Spatiotemporal Reuse and Resampling

##### Bitterli et al. 2020. Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting

- [DOI: 10.1145/3386569.3392481](https://doi.org/10.1145/3386569.3392481)
- Neighboring pixels와 previous frames의 light samples를 reservoir resampling으로 공유한다.
- 매우 많은 lights가 존재하는 환경에서도 적은 candidate samples로 direct lighting을 계산한다.
- Query 사이의 spatiotemporal correlation을 적극적으로 활용한다.


##### Ouyang et al. 2021. ReSTIR GI: Path Resampling for Real-Time Path Tracing

- [DOI: 10.1111/cgf.14378](https://doi.org/10.1111/cgf.14378)
- ReSTIR의 spatiotemporal resampling을 multi-bounce indirect-lighting paths로 확장한다.
- Important path information을 pixels와 frames 사이에서 공유한다.
- Light transport 계산 자체를 제거하기보다 expensive samples의 활용도를 높인다.


##### Kettunen et al. 2023. Conditional Resampled Importance Sampling and ReSTIR

- [DOI: 10.1145/3610548.3618245](https://doi.org/10.1145/3610548.3618245)
- Resampled Importance Sampling을 conditional probability spaces로 확장한다.
- Path suffix와 같은 conditional substructure를 unbiased하게 재사용할 수 있는 조건을 분석한다.
- Full path가 아니라 path의 일부를 다른 query에서 재사용하는 수학적 기반을 제공한다.


## Neural Light Transport

##### Zhang et al. 2021. Neural Light Transport for Relighting and View Synthesis

- [DOI: 10.1145/3446328](https://doi.org/10.1145/3446328)
- Known geometry 위에서 scene light transport를 neural representation으로 학습한다.
- Physically based diffuse component와 learned residual transport를 결합한다.
- Relighting과 novel-view synthesis를 위해 reusable neural transport representation을 사용한다.


##### Sun et al. 2021. NeLF: Neural Light-Transport Field for Portrait View Synthesis and Relighting

- [DOI: 10.2312/sr.20211299](https://doi.org/10.2312/sr.20211299)
- 3D point에서 light-transport coefficients를 예측하는 Neural Light-Transport Field를 제안한다.
- 새로운 viewpoint와 environment lighting에 대해 portrait appearance를 합성한다.
- Scene의 relightable transport information을 neural field 형태로 표현한다.


##### Hadadan et al. 2021. Neural Radiosity

- [DOI: 10.1145/3478513.3480569](https://doi.org/10.1145/3478513.3480569)
- Neural network로 spatial-directional radiance function을 표현한다.
- Ground-truth radiance 없이 Rendering Equation residual을 최소화하여 학습한다.
- View-independent radiance solution을 얻지만 scene별 optimization이 필요하다.


##### Müller et al. 2021. Real-Time Neural Radiance Caching for Path Tracing

- [DOI: 10.1145/3450626.3459812](https://doi.org/10.1145/3450626.3459812)
- Rendering 중 생성되는 radiance samples를 작은 neural network에 online 학습한다.
- Path를 일정 지점에서 terminate하고 remaining radiance를 neural cache로 근사한다.
- Dynamic scene에 적응할 수 있지만 approximation에 따른 bias와 error가 존재한다.


##### Zeng et al. 2025. RenderFormer: Transformer-Based Neural Rendering of Triangle Meshes with Global Illumination

- [DOI: 10.1145/3721238.3730595](https://doi.org/10.1145/3721238.3730595)
- Triangle, material, lighting information에서 global illumination image를 직접 생성한다.
- View-independent transport stage와 view-dependent rendering stage를 분리한다.
- Unseen scene으로 generalize하지만 Transformer의 높은 computation cost와 scene-scale 제한이 존재한다.


##### Xu et al. 2026. A Generalizable Light Transport 3D Embedding for Global Illumination

- [DOI: 10.1145/3799902.3811095](https://doi.org/10.1145/3799902.3811095)
- Geometry와 material을 포함한 point cloud로부터 3D light transport embedding을 생성한다.
- Query point 주변의 embeddings를 이용해 diffuse global illumination을 예측한다.
- Per-scene retraining 없이 unseen scenes에 generalize하는 scene-space transport representation을 목표로 한다.


## Differentiable Light Transport

##### Nimier-David et al. 2020. Radiative Backpropagation: An Adjoint Method for Lightning-Fast Differentiable Rendering

- [DOI: 10.1145/3386569.3392406](https://doi.org/10.1145/3386569.3392406)
- Reverse-mode differentiation을 별도의 adjoint light transport 문제로 해석한다.
- 전체 forward rendering history를 저장하지 않고 primal과 adjoint simulations로 gradients를 계산한다.
- Light transport의 operator structure를 이용해 differentiable rendering computation을 재구성한다.