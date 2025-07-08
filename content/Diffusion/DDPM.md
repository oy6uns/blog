---
tags:
  - Diffusion
---

> [!idea] Idea
>작은 sequence에서의 확산은 forward와 reverse **모두 가우시안**일 수 있다. <br>
  작은 공기 입자의 다음 위치는 가우시안 분포 안에서 결정될 수 있다.

## forward process

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

결과적으로, 두 분산이 정확히 더해서 1이 되게 된다. 이 덕분에 **매 스텝마다 전체 분산이 일정하게 유지**된다!
![[스크린샷 2025-07-08 오후 12.56.48.png]]

## 예시
실제 예시를 토대로 좀 더 쉽게 이해해보자. 

forward process는 아래와 같이 나타낼 수 있다. 
$$
q(x_{1:T}|x_0) = \Pi_{t=1}^Tq(x_t|x_{t-1})
$$
에서 $q(x_t|x_{t-1})$는 다음과 같은 **가우시안 분포**이다:
$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t;\;\sqrt{1-\beta_t}x_{t-1}, \beta_tI) 
$$

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
$q(x_t\mid x_{t-1})$: 한 스텝마다 $u_t=\sqrt{1-\beta_t}x_{t-1}, \sigma_t^2=\beta_t$인 가우시안.<br><br>
**샘플링**:
$$
x_t = \mu_t + \sqrt{\beta_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,1)
$$
이를 $T$번 반복하면 $q(x_{1:T}\mid x_0)$가 정의된다. <br>
이렇게 <b><font color="#e36c09">“한 스텝당 평균과 분산을 고정된 스케줄대로 정하고, 거기에 픽셀 단위 가우시안 노이즈를 더하는 것”</font></b>이 바로 **DDPM forward process**이다. 
![[스크린샷 2025-07-08 오후 12.57.18.png]]
계속해서 노이즈를 더해나가서, 최종적으로 $q(x_T)$ 라는 [[Gaussian Noise]]가 나오게 된다. 

## Backward Process
이젠 노이즈를 걷어낼 차례이다. <br>
![[스크린샷 2025-07-08 오후 5.05.47.png]]
- 흰점 $x_t$는 **“지금 우리가 보고 있는 $t$단계 샘플”** 이다. 
- 녹색 곡선은 **“노이즈를 뿌린 뒤 $t-1$단계에서 나올 수 있는 값들의 전반적인 분포”** 이다. 

우리의 목표는 **“관측한 $x_t​$라는 값 뒤에 숨어 있던 $x_{t-1}$가 어떤 값이었을지”** **사후분포(posterior)** 를 추정하는 것이다. 

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
        **posterior** $q(x_{t-1}\mid x_t)$가  
        **더 뾰족하고 단일한 피크**를 가지게 돼서
    - <font color="#e36c09">역방향으로 “되돌리기”가 훨씬 <b>정확하고 안정적</b>이 되는 것이다!</font>

즉, **짧게, 세세하게** 여러 스텝을 거칠수록  
각 스텝의 노이즈 주입량이 작아져 분포가 좁아지고,  
결국 **“한 단계씩 복원” 하기가 쉬워진다**는 의미가 된다. 






#### References
1. https://www.youtube.com/watch?v=uFoGaIVHfoE
2. https://www.youtube.com/watch?v=fbLgFrlTnGU


