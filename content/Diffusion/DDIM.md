---
date: 2025-07-11
created: 2025-07-11
modified: 2025-07-16
tags:
  - Diffusion
conf:
  - ICLR
year:
  - "2021"
link: https://arxiv.org/pdf/2010.02502
---
> Denoising Diffusion Implicit Models (ICLR ‘21)
> https://arxiv.org/pdf/2010.02502

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
**기존에 학습된 Diffusion Model**을 그대로 가져와서 <b><font color="#e36c09">샘플링 시에만 Non-Markovian Sampling을 선택</font></b>하여 **훨씬 적은 단계로 빠르게 고품질의 이미지를 생성**하는 것이 가능하다!<br><br>
# forward process
## DDIM’s Non-Markovian Process
$$
q_\sigma(x_{1:T}\mid x_0)
\;:=\;
q_\sigma(x_T\mid x_0)\;\prod_{t=2}^{T}q_\sigma(x_{t-1}\mid x_t,\,x_0)
$$
결론부터 말하자면, <b><font color="#e36c09">위의 DDIM의 forward 식은 DDPM과 완전히 일치한다.</font></b><br>DDPM에서는 $q(x_{1:T}​∣x_0​)=\prod_{t=1}^T​q(x_t​∣x_{t−1}​)$ 을 통해 명시적으로 **“직전 상태만 보고 다음 상태를 만든다”** 는 Markovian 가정을 사용하였다. <br><br>
DDIM에서는 ”$\prod_{t=2}^{T}q_\sigma(x_{t-1}\mid x_t,\,x_0)$”의 **non-Markovian 항이 어떻게 나오게 되었는지를 유도**해보자. <br>
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
&= q(x_T\mid x_0)\;\prod_{t=2}^T q(x_{t-1}\mid x_t,\,x_0)
\end{aligned}
$$
맨 마지막에는 오직 $q(x_T\mid x_0)$ 와 $\prod_{t=2}^T q(x_{t-1}\mid x_t, x_0)$ 만이 남게된다. <br><br>**“원래 DDPM의 forward $q(x_{1:T}\mid x_0)=\prod_t q(x_t\mid x_{t-1})$ process”** 를 bayesian rule과 **Markovian 가정**으로 텔레스코핑(telescoping) 해 보면 $\prod_{t=2}^T q(x_{t-1}\mid x_t,\,x_0)$ 이라는 <b><font color="#e36c09">non-Markovian한 인자</font></b>를 얻을 수 있게 된다! <br>결과적으로, **DDIM의 forward process 식**은 **DDPM의 forward process 식과 동일하게 표현**될 수 있다는 것을 알 수 있다

# DDPM vs. DDIM
noise를 추가하는 **forward process는 동일**하다는 것을 알았으니, 이제 **sampling** 과정에서의 둘의 차이를 알아보자!

## 학습 과정 (Training, Loss 계산 시):
- 우리는 원본 이미지 **$x_0$​를 가지고 있다.** 따라서 $q(x_{t-1}|x_t, x_0)$라는 **'이상적인 정답' 분포**를 계산할 수 있다. 
- DDPM, DDIM의 궁극적인 목표는 우리 모델($p_θ​(x_{t−1}​∣x_t​)$)이 이 정답 분포($q$)를 최대한 똑같이 모방하도록 학습시키는 것이다. 

## 샘플링 (Sampling, 이미지 생성 시):
### 📌 DDPM
> **“학습할 때만 정답지를 보고, 시험 볼 땐 안 봐요”**

- 우리는 **$x_0$​가 없는 상태**에서 랜덤 노이즈 $x_T$​로부터 이미지를 만들어야 한다. 
- 이때 모델 $p_\theta(x_{t-1}|x_t)$는 정답지($x_0$) 없이, 오직 현재 상태 $x_t$​만 보고 "훈련받은 대로" 한 step씩 복원해가며 $x_{t-1}$을 생성한다. 
- $x_0$​를 모르기 때문에 이 과정은 순차적인 **Markovian**일 수밖에 없다.
#### 샘플링 수식
$$
x_{t−1}​∼\mathcal{N}(μ_θ​(x_t​,t),σ_t^2​I)
$$
여기서 평균 $\mu_\theta$는 다음과 같이 정의된다:
$$
μ_θ​(x_t​,t)=\frac{1}{\sqrt{α_t}}​​​\bigg(x_t​−\frac{1−α_t}{\sqrt{1−\bar{α}_t}}​​ϵ_θ​(x_t​,t)\bigg)
$$
그리고 분산 $σ_t^2$는 학습 시 고정된 스케줄에 따라 주어진다. 
즉, DDPM 샘플링은 매 스텝마다
1. 예측한 Noise $\epsilon_\theta$로부터 $μ_θ​(x_t​,t)$ 계산
2. $\mathcal{N}(μ_θ​,σ_t^2​I)$에서 <font color="#31859b">random sample</font> 뽑기
로 이루어진다. 

### 📌 DDIM
> **“시험 볼 때도 정답지를 예측해서 봐요”**

- DDIM은 노이즈를 예측($ϵ_θ​$)하는 모델을 이용해 **현재 $x_t$​로부터 온전한 $x_0$​를 먼저 만든다** (**이를 $\hat{x}_0$​라고 하자!)** <br>→ $x_t$에 예측한 노이즈 $\epsilon_\theta$를 빼주면 $\hat{x}_0$를 얻어낼 수 있다. 
- 그다음, **예측한​** $\hat{x}_0$를 '정답지'처럼 사용하여 $q(x_{t-1}|x_t, \hat{x}_0)$ 공식에 따라 $x_{t-1}$을 **결정론적으로 계산**해낸다. 
- $x_0$를 (비록 예측값 $\hat{x}_0$이지만) 직접적으로 사용하기 때문에 이 과정은 **Non-Markovian**이 된다!
#### 샘플링 수식
$$
\hat x_0
= \frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\alpha_t}}
\quad(\text{“예측된 }x_0\text{”})
$$
$$
\begin{aligned}
x_{t-1}
&=
\underbrace{\sqrt{\alpha_{t-1}}\,\hat x_0}
          _{\substack{\text{predicted }x_0\text{을}\\\text{t-1 시점으로 보낸 항}}}
\;+\;
\underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t, t)}
          _{\text{“}x_t\text{를 향하는 방향”}}
\;+\;
\underbrace{\sigma_t\,\epsilon_t}_{\text{random noise (보통 0)}}
\end{aligned}
$$
DDIM에서는 보통 $(\sigma_t = 0)$으로 두어
![[스크린샷 2025-07-16 오후 9.07.35.png]]
$$
x_{t-1}
= \sqrt{\alpha_{t-1}}\,\hat x_0
\;+\;
\sqrt{1-\alpha_{t-1}}\,\epsilon_\theta(x_t, t)
$$
처럼 결정론적으로 한 step씩 복원해나간다.<br><br><br>

## So what's the benefit?
그래서 non-Markovian 가정으로 sampling 시에 어떤 이점이 있는걸까? 
![[스크린샷 2025-07-15 오후 2.34.31.png]]
**DDPM(Figure 1 왼쪽)** 에서는 **Markovian Process 전제**로 하기 때문에 매 타임스텝 $t=T, T-1, …, 1$ 마다 차례대로 한 칸씩 denoising을 수행해야 해서 총 T번의 네트워크 호출이 필요하다. <br>하지만, **DDIM(Figure 1 오른쪽)** 에서는 reverse process에 $x_0$(*위에서 설명했듯 실제로는 예측한 noise를 통해 근사적으로 계산된 $\hat{x}_0$를 사용한다.*) 를 명시적으로 집어넣어 “$x_t→x_{t-1}$” posterior에 **$x_0$가 조건으로 항상 쓰이게 된다.** <br>
![[스크린샷 2025-07-15 오후 2.06.15.png]]
DDIM의 <b><font color="#e36c09">non-Markovian posterior</font></b>를 이용하면, 중간 스텝을 <b><font color="#e36c09">“건너뛰는”</font></b> accelerated generation이 가능해진다!<br><br>skip schedule $\tau$를 정의하고, 
$$
τ=[τ_N​,τ_{N−1}​,…,τ_0​],\quadτ_N​=T,τ_0​=0,τ_k​>τ_{k−1}​
$$
정해진 $\tau$에 따라 **한 번에 건너뛰는 denoising을 수행**한다. 
$$
x_{τ_{k−1}}​​∼p_θ​(x_{τ_{k−1}}​​∣x_{τ_k}​​)≈q(x_{τ_{k−1}}​​∣x_{τ_k}​​,x_0​)
$$
결과적으로, denoising 횟수가 $T \rightarrow N$ 로 줄어 속도는 $\frac{K}{N}$배 빨라지고, 샘플 품질은 DDPM 대비 큰 손실 없이 유지된다!<br>

## 중간 스텝을 어떻게 건너뛸 수 있는걸까?
전체 **스텝을 모두 거치지 않고 일부를 건너뛰기**만 해도, 원**본 DDPM과 동등한 성능을 유지**하면서 **훨씬 빠르게 작동**할 수 있다는 것이 가능한건가 싶다. 이번 단락에서는 위에서 다룬 내용들의 엄밀성을 차근차근 살펴보겠다. 

### 우선 다시 식으로 돌아가서, 
왜 DDIM paper에서 사용한 식에서
$$
q(x_T\mid x_0)\;\prod_{t=2}^T q(x_{t-1}\mid x_t,\,x_0)
$$
$q(x_{t-1}\mid x_t, x_0)$는 $x_t$와 $x_0$가 주어졌을 때, denoising 방향의 single step 이고, <br>bayesian rule에 따라 다음과 같이 풀어 쓸 수 있다. 
$$
q(x_{t-1}\mid x_t, x_0)
= \frac{q(x_t\mid x_{t-1},x_0)\;q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
$$<br>
$q(x_t\mid x_{t-1},x_0)$와 $q(x_t\mid x_0)$는 [[DDPM]]에서 아래와 같이 구하였다. 
$$
q(x_t\mid x_{t-1},x_0) := \mathcal{N}\bigg(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I}\bigg)
$$
$$
\begin{aligned}
q(x_t\mid x_0)=\mathcal{N}\bigg(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I}\bigg), \\ 
where \quad \alpha_t:=1-\beta_t \;and\; \bar{\alpha}_t:=\prod^t_{s=1}\alpha_s
\end{aligned}
$$




### References
1. https://arxiv.org/pdf/2010.02502
2. https://www.youtube.com/watch?v=uFoGaIVHfoE&t=3657s
3. https://www.youtube.com/watch?v=TscMZOf5gXg