---
date: 2025-07-09
created: 2025-07-09
modified: 2025-07-09
---

Likelihood는 ‘어떤 일이 일어날 가능성’을 나타낸다. <br>
더 정확히 말하면 관측된 사건이 고정된 상태에서, 확률 분포가 변할 때(=확률 분포를 모를 때 = 가정할 때), 확률을 표현하는 단어이다. <br><br>
즉, 어떠한 관측값 $x$이 주어질 때, 변화되는 확률 분포 $p(\theta)$에서 주어진 관측값이 나올 확률이다. 
$$
\mathcal{L}(\theta|x) = p_\theta(x)
$$
다르게 말하면, “**모델 $p_\theta$가 관측된 $x$를 얼마나 잘 설명하느냐”를 수치화**한 것으로 볼 수 있다. <br>
<br>![[스크린샷 2025-07-09 오후 1.54.22.png]]
그림과 같이 $x_0$를 관측할 확률 $p_\theta(x_0)$은, **이전 단계에서 $x_0$로 오는 모든 가능한 trajectory(from latent $z$)를 marginalize**해야지 구할 수 있다.
>왜 Generative Model은 $p_\theta(x)$를 배워야할까? [[Discrimininative vs. Generative]]

<br>그러나, **가능한 전 영역을 적분한다는 건 불가능**하다.
<br>
직접 계산하기 어려운 모델의 사후 분포 $p(z|x)$를 계산하기 편한 분포 $q(z|x)$로 근사해서 다룬다. 이를, <b><font color="#e36c09">Varaiational Inference</font></b>라고 한다. <br>
$p_\theta(x_0)$<b><font color="#e36c09">의 Lower Bound는 아래와 같이 Maximize한다. </font></b>

## ELBO (Evidence Lower Bound)
$$
log\;p_\theta(x) \ge variational \; lower \;bound
$$
1. 원래 적분 식에서
$$
\log p_\theta(x) = \log \int p_\theta(x,z)\,dz
$$
2. $q(z|x)$를 분모 분자에 곱해준 뒤 정리하고, 
$$
\int p_\theta(x,z)\,dz
= \int \frac{q(z\mid x)}{q(z\mid x)}\,p_\theta(x,z)\,dz
= \int q(z\mid x)\,\frac{p_\theta(x,z)}{q(z\mid x)}\,dz
$$
3. 어떤 함수 $f(z)$에 대해 $\int q(z)f(z)dz = \mathbb{E}_q[f(z)]$ 이므로, 
$$
\int q(z\mid x)\,\frac{p_\theta(x,z)}{q(z\mid x)}\,dz
= \mathbb{E}_{q(z\mid x)}\;\!\!\Bigl[\tfrac{p_\theta(x,z)}{q(z\mid x)}\Bigr]
$$
4. jensen 부등식을 적용하면, $log$를 기댓값 안으로 집어 넣을 수 있게 된다. 
$$
\log p_\theta(x)
= \log \mathbb{E}_{q(z\mid x)}\;\!\!\Bigl[\tfrac{p_\theta(x,z)}{q(z\mid x)}\Bigr]
\;\ge\;
\mathbb{E}_{q(z\mid x)}\;\!\!\Bigl[\log \tfrac{p_\theta(x,z)}{q(z\mid x)}\Bigr]
$$
$$
\mathbb{E}_{q(z\mid x)}[\log p_\theta(x,z)]
\;=\;
\mathbb{E}_{q(z\mid x)}[\log p_\theta(x\mid z)]
\;+\;
\mathbb{E}_{q(z\mid x)}[\log p(z)]
$$
5. **결과적으로 아래와 같이 $(A),(B)$ 두 개의 항으로 정리가 된다.** 
$$
\log p_\theta(x)
\;\ge\;
\underbrace{\mathbb{E}_{q(z\mid x)}\bigl[\log p_\theta(x\mid z)\bigr]}_{(A)}
\;-\;
\underbrace{D_{\mathrm{KL}}\bigl(q(z\mid x)\,\|\,p(z)\bigr)}_{(B)}.
$$
## (A), (B)가 갖는 의미
### (A): Reconstruction Loss
> reconstruction 성능을 높이는 항

$\mathbb{E}_{q(z\mid x)}\bigl[\log p_\theta(x\mid z)\bigr]$는
- “$q(z\mid x)$, 즉 우리가 뽑아온 $z$ 샘플들”을 디코더 $p_\theta(x\mid z)$에 넣었을 때
- **관측된 $x$를 얼마나 잘 재현하는지**(log-likelihood)를 평균 내 본 값이다. 

쉽게 말해, **“우리 샘플링 방식으로 뽑은 latent variable $z$로 모델이 $x$를 잘 생성하느냐”** 를 측정한다!

### (B): Regularization Loss
> 샘플링 분포가 사전분포를 벗어나지 않도록 견제하는 항

$D_{KL}​(q(z∣x)∥p(z))$는
- “우리 $q(z\mid x)$가 만들어 내는 잠재 분포”가
- “prior(사전분포) $p(z)$”와 **얼마나 차이나는지**를 재는 척도이다. 
- 값이 커지면 “$q$가 prior 밖으로 너무 튀어 나갔다”는 뜻이니,
- 이 항을 최소화함으로써 **“$q$가 prior 범위 안에서 안정적으로 머물도록”** 억제하는 역할을 한다!


## 다시 ELBO로 돌아가서,
ELBO 식에서 잠재 변수 $z$를 확산 과정의 $x_{1:T}$​로, 관측 변수 $x$를 $x_0$​로 치환하면 다음과 같이 쓸 수 있다.
$$
\mathbb{E}_{\,q(x_{1:T}\mid x_0)}\bigl[\log p_\theta(x_0\mid x_{1:T})\bigr]
\;-\;
D_{\mathrm{KL}}\!\bigl(q(x_{1:T}\mid x_0)\,\|\,p_\theta(x_{1:T})\bigr)
$$
앞의 기댓값을 $\mathbb{E}_q$​로 간략히 표기하고, 뒤 항을 KL Divergence 형태의 기댓값으로 풀어쓰면,
$$
\mathbb{E}_q\bigl[\log p_\theta(x_0 \mid x_{1:T})\bigr]
\;-\;
\mathbb{E}_q\!\Bigl[\log \frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{1:T})}\Bigr]
$$
$$
\mathbb{E}_q\bigl[\log p_\theta(x_0 \mid x_{1:T})\bigr]
\;+\;
\log \frac{p_\theta(x_{1:T})}{q(x_{1:T}\mid x_0)}
$$
$$
=
\mathbb{E}_q\;\!\Bigl[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Bigr]
$$

이후 논문 수식 (21) 까지 유도는 [수식 정리 링크](https://www.notion.so/Diffusion-22bc798859de8046ac19d6da3288e984?source=copy_link) 참고 <br><br>

결과적으로 
$$
\mathbb{E}_q\Bigl[
-\,D_{\mathrm{KL}}\bigl(q(x_T\mid x_0)\,\|\,p(x_T)\bigr)
\;-\;
\sum_{t>1}D_{\mathrm{KL}}\bigl(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\bigr)
\;+\;
\log p_\theta(x_0\mid x_1)
\Bigr]
$$
위와 같은 수식으로 정리된다. 

이 값을 Maximize하는 것이 최종 목표이다. <br>DDPM에서는 이 <font color="#e36c09">ELBO의 음수(-ELBO)를 loss</font>로 보고, <font color="#e36c09">이를 Minimize하는 것을 목표</font>로 한다. 

![[스크린샷 2025-07-10 오후 3.13.16.png]]



### References
1. https://xoft.tistory.com/30
2. https://www.youtube.com/watch?v=fbLgFrlTnGU
3. https://heygeronimo.tistory.com/39
4. https://viplab.snu.ac.kr/viplab/courses/mlvu_2025_1/index.html