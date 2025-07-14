---
date: 2025-07-11
created: 2025-07-11
modified: 2025-07-11
tags:
  - Diffusion
---

> [!check] Title
> 매 time step $t$마다 sampling을 수행하는 것은 지나치게 많은 시간이 소요된다.<br>”DDPM이 가정한 마르코프 과정을 <b><font color="#e36c09">Non-Markovian Diffusion Process(비마르코프 확산)으로 대체</font></b>해서 **sampling 속도를 더 빠르게** 해보자!” 가 DDIM의 Novelty이다. 

DDIM은 기존 DDPM이 가정한 ‘**마르코프(Markovian) Diffusion(Forward) Process**’ 대신, 과거 상태 $x_{t-1}$뿐 아니라 원본 $x_0$ 정보까지 활용하는 ‘**비마르코프(non-Markovian) Diffusion Process**’와 그에 대응하는 **Reverse Process**을 새롭게 정의한다. <br><br>그러나, DDIM은 원래 DDPM에서 쓰던 **surrogate objective**(간소화된 손실 함수)를 그대로 최적화할 수 있도록 고안되어 
> 자세한 DDPM Loss식에 관한 설명은 [[ELBO (Evidence Lower Bound)]] 참고

$$
L_{\mathrm{simple}}(\theta)
= \mathbb{E}_{t, x_0, \epsilon}
\left\|
\epsilon \;-\;
\epsilon_{\theta}\bigl(\sqrt{\bar\alpha_t}\,x_0 \;+\;\sqrt{1-\bar\alpha_t}\,\epsilon,\;t\bigr)
\right\|^2
$$
**기존에 학습된 Diffusion Model**을 그대로 가져와서 <b><font color="#e36c09">샘플링 시에만 Non-Markovian Sampling을 선택</font></b>하여 **훨씬 적은 단계로 빠르게 고품질의 이미지를 생성**하는 것이 가능하다!


