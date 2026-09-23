# EXP-01: 1-spp Path Tracing Failure Visualization
date: 2026-09-23

## 1. Cornell-Box 

![](./cornell-box.png)  

Cornell Box에서는 1 spp로 렌더링했을 때 세 seed 모두 장면 전체에 강한 Monte Carlo noise가 나타났다. 이건 예상했던 결과로, sample 수가 1개뿐이라 Path Tracing의 variance가 충분히 줄어들지 못했기 때문이다. 그런데 absolute error map을 보면 한 가지 특이한 점이 있다. 천장의 area light와 가까운 위쪽 벽으로 갈수록 error가 더 크게 나타난다. 다만 이걸 바로 광원에 가까울수록 variance가 커진다고 해석하기는 어렵다. 단순히 그 영역의 radiance 자체가 높아서 absolute error도 같이 커졌을 가능성이 있기 때문이다. 그래서 다음 실험에서는 이 현상이 단순히 밝은 영역이라 error가 커 보이는 것인지, 아니면 실제로 light 근처에서 sampling variance가 더 커지는 것인지 분리해서 확인할 필요가 있다.


## 2. glass-of-water

![](./glass-of-water.png)

해당 이미지는 path tracing 에서 가장 못하는 장면이다. 즉, reflection과 refraction 에 대한 light 는 처리하지 못한다. 이는 computer graphics 에서 엄청 유명한 문제이다. 하지만 여기에서도 문제로 발견이 되는게 빛의 영향이 강하면 강할수록 많은  높은 error value 를 보여준다. 확실히 뭔가 문제가 있는 부분들이다. 여기에서는 specular manifold sampling, bidirectional path tracing techniques을 읽어보면 된다.

## 3. staircase2 

![](./staircase2.png)
