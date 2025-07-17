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
> Denoising Diffusion Implicit Models (ICLR ‘21)<br>
> https://arxiv.org/pdf/2010.02502

> [!check] Title
> 매 time step $t$마다 sampling을 수행하는 것은 지나치게 많은 시간이 소요된다.<br>”DDPM이 가정한 마르코프 과정을 <b><font color="#e36c09">Non-Markovian Diffusion Process(비마르코프 확산)으로 대체</font></b>해서 **sampling 속도를 더 빠르게** 해보자!” 가 DDIM의 Novelty이다. 

DDIM은 기존 DDPM이 가정한 ‘**마르코프(Markovian) Diffusion(Forward) Process**’ 대신, <br>과거 상태 $x_{t-1}$뿐 아니라 원본 $x_0$ 정보까지 활용하는 ‘**비마르코프(non-Markovian) Diffusion Process**’와 그에 대응하는 **Reverse Process**을 새롭게 정의한다. <br><br>DDIM은 원래 DDPM에서 쓰던 **surrogate objective**(간소화된 손실 함수)를 그대로 최적화할 수 있도록 고안되어 
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

^86dd67

맨 마지막에는 오직 $q(x_T\mid x_0)$ 와 $\prod_{t=2}^T q(x_{t-1}\mid x_t, x_0)$ 만이 남게된다. <br><br>**“원래 DDPM의 forward $q(x_{1:T}\mid x_0)=\prod_t q(x_t\mid x_{t-1})$ process”** 를 bayesian rule과 **Markovian 가정**으로 텔레스코핑(telescoping) 해 보면 $\prod_{t=2}^T q(x_{t-1}\mid x_t,\,x_0)$ 이라는 <b><font color="#e36c09">non-Markovian한 인자</font></b>를 얻을 수 있게 된다! <br>또한, **DDIM의 forward process 식**은 **DDPM의 forward process 식과 동일하게 표현**될 수 있다는 것도 알 수 있다. ^4e02d1

## ✅ forward 과정 수식 요약
### 1. forward 과정
$$
q(x_t \mid x_{t-1})
= \mathcal{N}\bigl(\sqrt{1-\beta_t}\,x_{t-1},\;\beta_t\,I\bigr)
$$
$$
\begin{align}
q(x_t \mid x_0)
= \mathcal{N}\bigl(\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)\,I\bigr), \quad\bar\alpha_t = \prod_{s=1}^t (1-\beta_s)
\end{align}
$$
### 2. Posterior 구하기 
Posterior를 Bayes 법칙과 가우시안 연산을 통해 유도하면,
$$
q(x_{t-1}\mid x_t,x_0)
= \mathcal{N}\bigl(x_{t-1};\;\mu_q,\;\Sigma_q\bigr)
$$
이고, 그 평균 ($\mu_q$)는
$$
\mu_q
= \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0
\;+\;
\frac{\sqrt{1-\beta_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t
$$
로 쓸 수 있다. 이를 다음과 같이 정리할 수도 있다:
$$
\mu_q
= \sqrt{\bar\alpha_{t-1}}\,x_0
\;+\;
\sqrt{\,1-\bar\alpha_{t-1}-\sigma_t^2\,}\;\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}}, \quad \sigma_t^2 = \frac{(1-\bar\alpha_{t-1})\,\beta_t}{1-\bar\alpha_t}
$$

# DDPM vs. DDIM
noise를 추가하는 **forward process는 동일**하다는 것을 알았으니, 이제 **sampling** 과정에서의 둘의 차이를 알아보자!

## 학습 과정 (Training, Loss 계산 시):
DDIM의 학습 또한 DDPM과 동일하다. 
- 우리는 원본 이미지 **$x_0$​를 가지고 있다.** 따라서 $q(x_{t-1}|x_t, x_0)$라는 **'이상적인 정답' 분포**를 계산할 수 있다. 
- DDPM, DDIM의 궁극적인 목표는 우리 모델($p_θ​(x_{t−1}​∣x_t​)$)이 이 정답 분포($q$)를 최대한 똑같이 모방하도록 학습시키는 것이다. 

## 샘플링 (Sampling, 이미지 생성 시):
### 📌 DDPM
> **“학습할 때만 정답지를 보고, 시험 볼 땐 안 봐요”**

우리는 **$x_0$​가 없는 상태**에서 랜덤 노이즈 $x_T$​로부터 이미지를 만들어야 한다. 이때 모델 $p_\theta(x_{t-1}|x_t)$는 정답지($x_0$) 없이, 오직 현재 상태 $x_t$​만 보고 "훈련받은 대로" 한 step씩 복원해가며 $x_{t-1}$을 생성한다. <br>
$x_0$​를 모르기 때문에 이 과정은 순차적인 **Markovian**일 수밖에 없다.
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

#### 1. 원본 $x_0$ 복원하기 
DDIM은 먼저 모델이 예측한 **노이즈 $ϵ_θ(x_t, t)​$로부터 “지금 이 시점의 이미지 $x_t$”가 어떤 원본 $x_0$에서 온 것인지를 역추정한다.** 
$$
\hat x_0
= \frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\alpha_t}}
\quad(\text{여기서 }\bar{\alpha}_t=\prod^t_{s=1}\alpha_s)
$$
- $\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ 의 선형 결합으로 $x_t$​가 만들어졌다는 forward 식을 재구성 한 것이다.
- 이 값이 **“예측된 원본”** $\hat{x}_0$이다. 
#### 2. 샘플링
> skip step 내용은 [아래 단락](DDIM#^58a6d6)에서 다룸

이제 진짜 posterior
$$
   q\bigl(x_{t-1}\mid x_t,x_0\bigr)
   =\mathcal{N}\bigl(\underbrace{\mu_q}_{\text{posterior mean}},\;\Sigma_q\bigr)   
$$
의 **평균 $\mu_q$** 만 써서, $x_{t–1}$을 직접 계산한다. <br>그 평균은 수학적으로 다음과 같이 깔끔히 정리된다. 
$$
   \mu_q
   =\sqrt{\bar\alpha_{t-1}}\,x_0
   \;+\;
   \sqrt{\,1-\bar\alpha_{t-1}-\sigma_t^2\,}\;\epsilon_\theta(x_t,t)
   $$
실제 inference(sampling)에서는 **“진짜 $x_0$”** 대신 **“예측된 $\hat{x}_0$”** 를 넣어, 
$$
x_{t-1}
= \underbrace{\sqrt{\bar\alpha_{t-1}}\,\hat x_0}_{\substack{\text{한 스텝 전 분포로}\\\text{되돌린 예측 원본}}}
\;+\;
\underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\;\epsilon_\theta(x_t,t)}_{\substack{\text{남아 있는 노이즈 성분을}\\\text{적절히 반영하는 항}}}
$$
- 첫 번째 항은 “모델이 생각하는 clean image($\hat x_0$)를 <br>바로 $t-1$ 정도의 노이즈 수준으로 되돌린” 부분이고,
- 두 번째 항은 “원래 $x_t$​의 노이즈 방향($\epsilon_\theta$​)을<br>부족하지 않게 섞어 줘서 자연스러운 transition을 만드는” 부분이다. 

DDIM에서는 보통 $(\sigma_t = 0)$으로 두어
![[스크린샷 2025-07-16 오후 9.07.35.png]]
$$
x_{t-1}
= \sqrt{\alpha_{t-1}}\,\hat x_0
\;+\;
\sqrt{1-\alpha_{t-1}}\,\epsilon_\theta(x_t, t)
$$
처럼 결정론적으로 한 step씩 복원해나간다.
<br><br>

## So what's the benefit?

^58a6d6

그래서 <b><font color="#e36c09">non-Markovian 가정으로 sampling 시에 어떤 이점</font></b>이 있는걸까? <br>
위에서 설명했던 DDIM의 sampling 과정을 다시 살펴보자. 

**DDPM** 에서는 **Markovian Process 전제**로 하기 때문에 매 타임스텝 $t=T, T-1, …, 1$ 마다 차례대로 한 칸씩 denoising을 수행해야 해서 총 T번의 네트워크 호출이 필요했다. 
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
하지만, **DDIM(위의 샘플링 식)** 에서는 reverse process에 $x_0$(*위에서 설명했듯 실제로는 예측한 noise를 통해 근사적으로 계산된 $\hat{x}_0$를 사용한다.*) 를 명시적으로 집어넣어 “$x_t→x_{t-1}$” posterior에 **$x_0$가 조건으로 항상 쓰이게 된다.** <br>
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
전체 **스텝을 모두 거치지 않고 일부를 건너뛰기**만 해도, **원본 DDPM과 동등한 성능을 유지**하면서 **훨씬 빠르게 작동**할 수 있다는 것이 가능한건가 싶다.<br>아까 [DDIM’s non-Markovian Forward Process](DDIM#^4e02d1) 에서 우리는 **DDIM의 모든 forward 과정이 모든 $t$에 대해 가우시안으로 명시**돼 있기 때문에, 이론상으로는 어떤 시점 $t$에 대해서도 posterior
$$
q(x_t\mid x_{t-1},x_0) = \mathcal{N}\bigg(\mathbf{x}_{t-1};\;\mu_q(x_t, x_0), \;\Sigma_q\bigg)
$$
의 평균 $\mu_q$와 분산 $\Sigma_q$를 정확히 계산할 수 있다. <br><br>따라서 이론적으로는 $x_t$와 진짜 $x_0$가 주어지면, 모든 $t$에 대해 이 posterior를 **closed-form**으로 구할 수 있다. <br><br>다만 실제 샘플링(inference) 단계에서는 진짜 $x_0$를 모르기 때문에, <br>**네트워크로 예측한 $\hat{x}_0$을 넣어서**
$$
x_{t-1}=\tilde{\mu}_t(x_t, \hat{x}_0)
$$
형태로 **근사하여 사용**한다는 점만 기억하면 된다!


<br><br><br>

### References
1. https://arxiv.org/pdf/2010.02502
2. https://www.youtube.com/watch?v=uFoGaIVHfoE&t=3657s
3. https://www.youtube.com/watch?v=TscMZOf5gXg