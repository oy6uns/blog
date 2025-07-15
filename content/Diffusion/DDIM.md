---
date: 2025-07-11
created: 2025-07-11
modified: 2025-07-11
tags:
  - Diffusion
conf:
  - ICLR
year:
  - "2021"
link: https://arxiv.org/pdf/2010.02502
---

> [!check] Title
> 매 time step $t$마다 sampling을 수행하는 것은 지나치게 많은 시간이 소요된다.<br>”DDPM이 가정한 마르코프 과정을 <b><font color="#e36c09">Non-Markovian Diffusion Process(비마르코프 확산)으로 대체</font></b>해서 **sampling 속도를 더 빠르게** 해보자!” 가 DDIM의 Novelty이다. 

DDIM은 기존 DDPM이 가정한 ‘**마르코프(Markovian) Diffusion(Forward) Process**’ 대신, 과거 상태 $x_{t-1}$뿐 아니라 원본 $x_0$ 정보까지 활용하는 ‘**비마르코프(non-Markovian) Diffusion Process**’와 그에 대응하는 **Reverse Process**을 새롭게 정의한다. <br><br>DDIM은 원래 DDPM에서 쓰던 **surrogate objective**(간소화된 손실 함수)를 그대로 최적화할 수 있도록 고안되어 
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

# forward process
## DDIM’s Non-Markovian Process
$$
q_\sigma(x_{1:T}\mid x_0)
\;:=\;
q_\sigma(x_T\mid x_0)\;\prod_{t=2}^{T}q_\sigma(x_{t-1}\mid x_t,\,x_0)
$$
위의 수식이 어떻게 나오는지 전개해보자. <br>

### step-by-step
한 step을 bayes rule로 나타내면, 
$$
q(x_{t-1}\mid x_t, x_0)
= \frac{q(x_t\mid x_{t-1},x_0)\;q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
= \frac{q(x_t\mid x_{t-1})\;q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)} \quad (*)
$$

> [!question] (*)은 Markovian을 가정하고 전개한거 아닌가요?
> DDIM에서 “non-Markovian”이라 부르는 건 **reverse 수식의 인자에 $x_0$를 끌어다 쓰는 것 때문**이지, **forward 단계 자체가 Markovian 가정을 깨는 것은 아니다.**<br>**따라서, 두 번째 식에서 세 번째 식으로 전개되는 과정은 문제가 없다!**

t = 2일 때, 
$$
q(x_1\mid x_0)\;q(x_2\mid x_1)
= q(x_2\mid x_0)\;q(x_1\mid x_2, x_0)
$$
t = 3일 때, 
$$
q(x_2\mid x_0)\;q(x_3\mid x_2)
= q(x_3\mid x_0)\;q(x_2\mid x_3, x_0)
$$

기존 DDPM의 forward process는 다음과 같았다: 
$$
q(x_{1:T}|x_0) = \Pi_{t=1}^Tq(x_t|x_{t-1})
$$
$(*)$를 한 step씩 적용해보면, 
$$
\begin{aligned}
\prod_{t=1}^T q(x_t\mid x_{t-1})
&= q(x_1\mid x_0)\,q(x_2\mid x_1)\,\dots\,q(x_T\mid x_{T-1})\\
&= \bigl[q(x_1\mid x_0)\,q(x_2\mid x_1)\bigr]\;q(x_3\mid x_2)\,\dots\,q(x_T\mid x_{T-1})\\
&= q(x_2\mid x_0)\,q(x_1\mid x_2,x_0)\,q(x_3\mid x_2)\,\dots\,q(x_T\mid x_{T-1})\\
&= q(x_3\mid x_0)\,q(x_2\mid x_3,x_0)\,q(x_1\mid x_2,x_0)\,\dots\,q(x_T\mid x_{T-1})\\
&\ \ \vdots\\
&= q(x_T\mid x_0)\;\prod_{t=2}^T q(x_{t-1}\mid x_t,\,x_0).
\end{aligned}
$$
맨 마지막에는 오직 $q(x_T\mid x_0)$ 와 $\prod_{t=2}^T q(x_{t-1}\mid x_t, x_0)$ 만이 남게된다. <br><br>결과적으로, **DDIM의 forward process 식**은 **DDPM의 forward process 식과 동일하게 표현**될 수 있다는 것을 알 수 있다. 

## So what's the benefit?
그래서 식을 위와 같이 바꾸면 어떤 이점이 있는걸까?
![[스크린샷 2025-07-15 오후 2.34.31.png]]
**DDPM(Figure 1 왼쪽)** 에서는 **Markovian Process 전제**로 하기 때문에 매 타임스텝 $t=T, T-1, …, 1$ 마다 차례대로 한 칸씩 denoising을 수행해야 해서 총 T번의 네트워크 호출이 필요하다. <br>하지만, **DDIM(Figure 1 오른쪽)** 에서는 reverse process에 $x_0$를 명시적으로 집어넣어 “$x_t→x_{t-1}$” posterior에 **$x_0$가 조건으로 항상 쓰이게 된다.** <br>
![[스크린샷 2025-07-15 오후 2.06.15.png]]
DDIM의 <b><font color="#e36c09">non-Markovian posterior</font></b>를 이용하면, 중간 스텝을 <b><font color="#e36c09">“건너뛰는”</font></b> accelerated generation이 가능해진다!<br><br>skip schedule $\tau$를 정의하고, 
$$
τ=[τ_K​,τ_{K−1}​,…,τ_0​],\quadτ_K​=T,τ_0​=0,τ_k​>τ_{k−1}​
$$
정해진 $\tau$에 따라 **한 번에 건너뛰는 denoising을 수행**한다. 
$$
x_{τ_{k−1}}​​∼p_θ​(x_{τ_{k−1}}​​∣x_{τ_k}​​)≈q(x_{τ_{k−1}}​​∣x_{τ_k}​​,x_0​)
$$
결과적으로, denoising 횟수가 $T \rightarrow K$ 로 줄어 속도는 $\frac{K}{T}$배 빨라지고, 샘플 품질은 DDPM 대비 큰 손실 없이 유지된다!

