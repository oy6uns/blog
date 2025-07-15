---
tags:
  - Diffusion
date: 2025-07-08
created: 2025-07-08
modified: 2025-07-08
conf:
  - NeurIPS
year:
  - "2020"
link: https://arxiv.org/pdf/2006.11239
---
> Denoising Diffusion Probabilistic Models (NeurIPS ‘20)
> https://arxiv.org/pdf/2006.11239


> [!idea] Idea
>작은 sequence에서의 확산은 forward와 reverse **모두 가우시안**일 수 있다. <br>
  작은 공기 입자의 이동을 상상해보면 작은 공기 입자의 **다음 위치는 가우시안 분포 안에서 결정**될 수 있다.

![[스크린샷 2025-07-08 오후 6.09.03.png]]
![[스크린샷 2025-07-08 오후 6.07.40.png]]
Diffusion은 크게 2가지(**forward process, backward process**)로 구성된다. <br> 
**forward process**는 노이즈를 점진적으로 추가해나가는 과정이고, <br>
**backward process**는 노이즈를 지워나가며 input으로 넣어준 이미지를 다시 복원해나가는 과정이다. <br>
위의 2번째 그림에 나와있듯이 노이즈를 추가하는 forward process는 단순히 노이즈를 이미지에 추가만 하면 되기에 따로 학습할 필요가 없고, 우리가 <b><font color="#e36c09">초점을 맞출 부분은 노이즈를 없애는 backward process(denoising)</font></b>이다. 

# Forward Process
> 노이즈를 점진적으로 추가해보자!

$$
\mathcal{N}(x_t;\;\sqrt{1-\beta_t}x_{t-1}, \beta_tI) 
$$
$$
x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon
$$

여기서
- $x_{t-1}$과 $\epsilon$은 독립이다.
- $Var(x_{t-1})=1, Var(\epsilon)=1$ 라 가정한다.

두 확률변수 $A$와 $B$가 독립이고, 상수 $a, b$ 가 있을 때 <br>
(공분산 항 $2ab\;Cov(A, B)$는 독립이면 0이 되기에)

$$
Var(aA+bB)=a^2Var(A)+b^2Var(B)
$$
$$
Var(x_t)=(\sqrt{1-\beta_t})^2\cdot1+(\sqrt{\beta_t})^2\cdot1=(1-\beta_t)+\beta_t=1.
$$

결과적으로, 두 분산이 정확히 더해서 1이 되게 된다. 이 덕분에 노이즈를 추가하는 **매 스텝마다 전체 분산이 일정하게 유지**된다!
![[스크린샷 2025-07-08 오후 12.56.48.png]]

## 예시
실제 예시를 토대로 좀 더 쉽게 이해해보자. 

forward process는 아래와 같이 나타낼 수 있다. 
$$
q(x_{1:T}|x_0) = \Pi_{t=1}^Tq(x_t|x_{t-1})
$$
원본 데이터 $x_0$에 순차적으로 노이즈를 추가해 $x_1, x_2, \dots, x_T$​ 를 생성하는 전체 과정을 나타낸다. 매 스텝이 **마르코프 성질(현재 상태만 보고 다음 상태를 결정)** 을 가지므로, 전체 확률은 **각 단계 전이 확률의 곱으로 분해**된다. <br>
$q(x_t|x_{t-1})$는 다음과 같은 **가우시안 분포**를 따른다:
$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t;\;\sqrt{1-\beta_t}x_{t-1}, \beta_tI) 
$$
- 평균($\sqrt{1-\beta_t}\,x_{t-1}$)은 “이전 상태의 신호를 $\sqrt{1-\beta_t}$만큼 보존”    
- 분산($\beta_t$)은 “크기 $\beta_t$의 가우시안 노이즈 추가”
- 즉, **작은 $t$ 단계마다 점진적으로 가우시안 노이즈를 끼워 넣어** 데이터를 점차 무작위에 가깝게 오염시키는 것을 수식화한 것이다. 

### single-step
초기값을 $x_0=0.8$ 이라고 하자. <br><br>
**첫번째 노이징(t=1)**
- 스케줄에서 $\beta_1=0.1$이라 하면, $\alpha_1=1-\beta_1=0.9$
- 평균(mean)  
$$
\\mu_1=\sqrt{\alpha_1}x_0=\sqrt{0.9}​×0.8≈0.9499×0.8=0.7600
$$
- 분산(variance)
$$
 \sigma_1^2=\beta_1=0.1
$$
- 즉, 
$$
q(x_1|x_0)=\mathcal{N}(0.7600, 0.1)
$$
- 여기서 한 번 샘플링을 한다면, $\epsilon~\mathcal{N}(0, 1)$ 예를 들어 $\epsilon=0.5$를 뽑았을 때, 
$$
x_1=0.7600+\sqrt{0.1}\times0.5 = 0.7600+0.3162×0.5≈0.9181
$$

### multi-step
> 한번에 바로 $x_0\rightarrow x_2$로 가는 $q(x_2|x_0)$

$$
q(x_2​∣x_0​)=N(\sqrt{α_1​α_2}​​x_0​,1−α_1​α_2​)
$$
- 여기서 $\alpha_1\alpha_2=0.9\times0.8=0.72$
- 평균 $\sqrt{0.72}\times0.8\approx0.8485\times0.8=0.6788$
- 분산 $1-0.72=0.28$
- 즉, 
$$
q(x_2|x_0)=\mathcal{N}(0.6788, 0.28)
$$

### 정리하자면, 
$q(x_t\mid x_{t-1})$: 한 스텝마다 $u_t=\sqrt{1-\beta_t}x_{t-1}, \sigma_t^2=\beta_t$인 가우시안이고, <br><br>
**샘플링**
$$
x_t = \mu_t + \sqrt{\beta_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,1)
$$
위의 샘플링을 $T$번 반복하면 $q(x_{1:T}\mid x_0)$가 정의된다. <br>
이렇게 <b><font color="#e36c09">“한 스텝당 평균과 분산을 고정된 스케줄대로 정하고, 거기에 픽셀 단위 가우시안 노이즈를 더하는 것”</font></b><br>이 바로 **DDPM forward process**이다. 
![[스크린샷 2025-07-08 오후 12.57.18.png]]
계속해서 노이즈를 더해나가면, 최종적으로 $q(x_T)$ 라는 [[Gaussian Noise]]가 나오게 된다. 

# Backward Process
>이젠 노이즈를 걷어낼 차례이다! 

![[스크린샷 2025-07-08 오후 5.05.47.png]]
- 흰점 $x_t$는 **“지금 우리가 보고 있는 노이즈가 t번 추가된 $t$단계 샘플”** 이다. 
- 녹색 곡선은 **“노이즈를 뿌린 뒤 $t-1$단계에서 나올 수 있는 값들의 전반적인 분포”** 이다. 

우리의 목표는 **“관측한 $x_t​$의 이전 step $x_{t-1}$가 어떤 값이었을지”** **사후분포(posterior)** $q(x_{t-1}|x_t)$를 추정하는 것이다. 

![[스크린샷 2025-07-08 오후 5.45.08.png]]
왜 time step을 작게 가져가야 할까?
- **시간(step) 간격이 작다**는 건, 각 단계에서 섞는 노이즈 $\beta_t$​가 **매우 작다**는 뜻이다. 
- 그러면

$$
q(x_t\mid x_{t-1})=\mathcal N\bigl(\sqrt{1-\beta_t}\,x_{t-1},\;\beta_tI\bigr)
$$
의 분산 $\beta_t$가 작아져서, 왼쪽 초록 그래프처럼 **매우 좁고 뾰족한 분포**가 된다. 
- 이렇게 분포가 좁으면
    - “$x_t$가 이 흰 점에 관측됐을 때,  
        이전 값 $x_{t-1}$는 어디였을까?”를 구하는  
        **posterior** $q(x_{t-1}\mid x_t)$가 **더 뾰족하고 단일한 피크**를 가지게 돼서
    - <font color="#e36c09">역방향으로 “되돌리기”가 훨씬 <b>정확하고 안정적</b>이 되는 것이다!</font>

즉, **짧게, 세세하게** 여러 스텝을 거칠수록  
각 스텝의 노이즈 주입량이 작아져 분포가 좁아지고,  
결국 **“한 단계씩 복원” 하기가 쉬워진다**는 의미가 된다. <br>

# Training
**2015년에 나왔던 [Diffusion Model](https://arxiv.org/pdf/1503.03585)** 과 **2020년에 나온 DDPM의 차이**를 들자면, <br>우선은 둘 다 본질적으로 **같은 목표**(<b><font color="#e36c09">각 단계의 reverse process</font></b> $p_θ(x_{t−1}​∣x_t​)$가 <b><font color="#e36c09">진짜 posterior</font></b> $q(x_{t-1}|x_t, x_0)$를 <b><font color="#e36c09">잘 근사하도록</font></b> 학습)
$$
\sum_{t=1}^T​D_{KL}​(q(x_{t−1}​∣x_t​,x_0​)∥p_θ​(x_{t−1}​∣x_t​))
$$
를 갖고 있지만, <b>parameterization</b>과 <b>loss function의 형태</b>에서 차이가 존재한다. 
### 1. Diffusion Model(2015)
[Diffusion Model](https://arxiv.org/pdf/1503.03585)에서는 $p_\theta$의 **평균 $\mu_\theta(\mathbf{x}_t, t)$과 분산 $\Sigma_\theta(\mathbf{x}_t, t)$을 직접 예측**하도록 설계되었다. 
### 2. DDPM(2020)
같은 KL 목표를 가지지만, <b><font color="#e36c09">reparameterization</font></b>을 진행하였다. 
1. Forward를 “노이즈 $\epsilon$”에 대한 관점으로 바꿈
$$
x_t​=\sqrt{\bar{α}_t}​​x_0​+\sqrt{1−\bar{α}_t​}​ϵ,\quadϵ∼N(0,I)
$$
2. $p_\theta$의 평균 $\mu_\theta(\mathbf{x}_t, t)$ 대신, noise $\epsilon$을 직접 예측하도록 네트워크 $\epsilon_\theta(\mathbf{x}_t, t)$를 학습
3. 손실을 단순한 **MSE**로 정리 
$$
\mathbb{E}\big[∥ϵ−ϵ_θ​(x_t​,t)∥^2\big]
$$
→ 결과적으로 2015년 논문과 Posterior matching을 사용한 것은 완전히 동일하지만,  
DDPM은 posterior 유도 과정을 “노이즈 예측”이라는 한 줄 MSE로 압축하여  
**학습 안정성과 구현 편의성을 크게 개선한 것**이라 할 수 있다!<br>


# Overall Step
결과적으로, 
![[스크린샷 2025-07-08 오후 5.53.44.png]]
input 확률 분포 $q(x_0)$ (그림의 파란색 영역) 로 부터 
> 확률 분포에 관한 내용은 [[Gaussian Noise]] 참고

<font color="#e36c09">noise를 추가해가고(forward), noise를 다시 제거(backward)하는 과정을 통해 </font><br>
기존의 확률 분포 $q(x_0)$로 돌아오는 것이최종 Goal이다. <br><br>
그러면 잘 돌아오게끔 어떻게 유도할 수 있을까?
> [[ELBO (Evidence Lower Bound)]]





#### References
1. https://www.youtube.com/watch?v=uFoGaIVHfoE
2. https://www.youtube.com/watch?v=fbLgFrlTnGU
3. https://wikidocs.net/275557
4. https://www.youtube.com/watch?v=zcEe78I_4TU


