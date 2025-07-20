---
date: 2025-07-18
created: 2025-07-18
modified: 2025-07-18
tags:
---
[[DDPM]] 내용을 안다는 가정 하에 아래 다양한 Diffusion 논문들은 핵심들만 짚어서 정리하였다. <br><br>
# Classifier Guidance 
> Diffusion Models Beat GANs on Image Synthesis (NeurIPS workshop ‘21)<br>https://arxiv.org/pdf/2105.05233

다양한 Diffusion Model들이 발표되었지만, GAN과 비교했을 때 이들 방법의 공통적인 한계는 **Condition을 반영할 수 없다는 것**이었다. (원하는 방향으로 학습시키는 것이 어려움)<br>Classifier Guidance에서는 Class 정보를 Condition으로 입력 받아 <b><font color="#e36c09">해당 Class 이미지만 생성할 수 있는 Diffusion Model을 제안</font></b>한다. **별도의 Classifier를 학습시켜 이로부터 나오는 Gradient를 활용**하기 때문에 Classifier Guidance라는 별칭이 붙었다. <br><br>두 가지 Contribution이 있다. 
1. Model Architecture Improvements
2. **Classifier Guidance** ✨

## Model Architecture Improvements
- 모델 크기(파라미터 수)는 일정하게 유지하며, 깊이(depth) 대비 너비(width)를 늘리기
- Attention Heads 수 증가
- **Multi-resolution Attention**: 16x16에만 적용하던 Attention을, $32\times32$, $16\times16$, $8\times8$ 해상도 전반에 확대 적용
- Rescale Residual Connections: Residual 연결 시에 output에 $\frac{1}{\sqrt{2}}$를 적용하여 분산을 맞춰줌<br>Scale 맞춰주면 합연산 후에도 분산이 보존되어 분산의 크기가 일정하게 유지된다. 
$$
y=\frac{1}{\sqrt{2}}(x+\mathcal{F}(x))
$$
## Classifier Guidance ✨
### Idea
1. pretrained된 Diffusion Model $p_\theta(x_{t-1}|x_t)$에, 
2. **노이즈가 섞인 이미지** $x_t$를 받아 “이 이미지가 클래스 $y$일 확률” $p_ϕ​(y|x_t)$을 예측하는 **Classifier를 추가로 학습**한 뒤, 
3. 두 확률을 곱해 “클래스 $y$를 만족하는 역확산 분포”를 정의한다. 
$$
p_{θ,ϕ}​(x_{t−1}​∣x_t​,y)∝p_θ​(x_{t−1}​∣x_t​)×p_ϕ​(y∣x_{t−1}​)
$$
4. 이 분포에서 샘플링하면, “일반적인 역확산”에 비해<br>$p_ϕ​(y\;|\;\cdot\;)$ 항이 클래스 $y$ 방향으로 샘플을 유도하는 역할을 해주어, 
5. 결과적으로 “조건부 (class-conditional) 생성”이 가능해진다. 

결과적으로, 새로운 평균은
$$
\tilde{\mu}=\underbrace{\mu(x_t, t)}_{\text{원래 모델이 예측한 평균}} + \underbrace{\Sigma(x_t, t)\nabla_{x_{t-1}}log\;p_ϕ​(y\;|\;x_{t-1})}_{\text{＂y일 확률＂을 높이는 방향으로의 한 걸음}}
$$
분산 $\Sigma$는 그대로 두고, 평균만 이동시킨 뒤<br>$\mathcal{N}(\tilde{\mu}, \Sigma)$에서 샘플을 뽑으면, <br>**”클래스 $y$ 쪽으로 가중된” Conditional Reverse Process**가 된다!<br><br>


# Classifier-free Guidance
> Classifier-Free Diffusion Guidance (ICLR’ 22)<br>https://arxiv.org/pdf/2207.12598

Classifier를 추가적으로 학습해야 한다는 단점을 극복하기 위해 나온 논문이다. <br><br>Classifier Guidance는 
1. **추가 Classifier를 학습**해야 하고, 
2. 그 Classifier는 노이즈가 섞인 이미지로 학습해야 하므로 **Pretrained된 일반적인 Classifier를 바로 사용할 수 없으며**, 
3. 확산 모델의 본래 스코어(“데이터 분포에서 자연스러운 이미지 방향”)와 Classifier Gradient(“분류기에는 $y$로 보이는 방향”)는 항상 일치하지 않는다. 따라서, 정도를 잘못 조절하면 모델이 학습한 **“자연스러운 분포”** 에서 꽤 벗어난 지점까지 샘플이 이동하게 될 수 있다. <br>
는 단점들이 존재한다. 

<br>이 논문의 핵심은 **Classifier 없이 Diffusion 모델을 Guidance**할 수 있는 방법을 제시한다. 따라서 **“Classifier-free Guidance”** 라는 이름으로 더 널리 알려져있다. 

본 논문은 단순히 샘플이 품질을 SOTA로 올리는 것에 집중하기 보다, **classifier-free guidance가 classifier guidance와 비슷하게 FID/ID를 달성할 수 있는지**를 보이고, classifier-free guidance를 이해하는 것을 목표로 한다. <br><br>

## Classifier-free training
![[스크린샷 2025-07-20 오후 6.40.57.png]]
원본 이미지 $x$와, 그 이미지의 레이블(class) $c$를 데이터셋에서 꺼내온다. <br><br>
**일정 확률 $p_{uncond}$에 따라, “아예 아무 레이블도 주지 않고($c\leftarrow \varnothing$)” 학습**한다. 이렇게 하면 모델이
- **레이블이 있을 때**는 <b><font color="#de7802">“고양이면 고양이, 개면 개” 처럼 조건부로 배우고</font></b>, 
- **레이블이 없을 때**는 그냥 <b><font color="#de7802">“있는 그대로의 자연스러운 이미지도 배우게 된다.”</font></b>


모델은 
- **노이즈가 섞인 $z_\lambda$** 와 (있다면) **레이블 $c$를 입력**으로 받아 
- 원래 이미지에 섞인 노이즈 양 $\hat{\epsilon}=ϵ_θ​(z_λ​,c)$를 예측한다. 
- DDPM과 마찬가지로, $∇_θ ‖ \epsilon_θ(z_λ, c) − \epsilon‖^2$ 노이즈 오차를 통해 $\theta$를 조금씩 업데이트한다. 
<br>
## Sampling
![[스크린샷 2025-07-20 오후 7.45.04.png]]
1. 훈련이 끝나면, 순수 가우시안 노이즈 $z_1 \sim \mathcal{N}(0, I)$를 뽑고
2. 매 denoising 스텝마다 <b><font color="#de7802">꺼낼 노이즈량</font></b> $\tilde{\epsilon_t}$을 정한다. <br>이는 조건부, 무조건부 Score를 섞어서 계산하면 된다. 
   - **조건부 Score**: $\epsilon_+=\epsilon_\theta(z_t, c)$ → <b><font color="#de7802">label이 c일 때 이렇게 노이즈를 빼라!</font></b>
   - **무조건부 Score**: $\epsilon_0=\epsilon_\theta(z_t)$ → <b><font color="#de7802">아무 조건 없을 때 이렇게 노이즈를 빼라!</font></b>
$$
\tilde{\epsilon_t}=(1+w)\epsilon_+\;-\;w\;\epsilon_0
$$

**샘플링할 때 별도의 Classifier를 쓰지 않고도**, <br>제거할 노이즈 값을 $ε_θ(z,c)$와 $ε_θ(z)$의 가중합으로만 정해도<br><b><font color="#de7802">원하는 클래스</font></b> $c$ <b><font color="#de7802">방향으로 denosing을 유도할 수 있게 된다!</font></b>
<br>![[스크린샷 2025-07-20 오후 8.39.44.png]]
위 그림과 같이 **$w$ 값을 키울수록 원하는 클래스에 더욱 가깝고 뚜렷한 샘플이 점점 더 많이 생성**되는 것을 확인할 수 있다. <br><br>

## Discussion
추가적인 Classifier의 log-prob Gradient를 쓰는 **Classifier-Guidance와 달리** <br>Classifier-free Guidance는 **순수 생성모델의 Score 두 개를 섞은 비보존(non-conservative) 벡터장**이 된다.

> [!question] non-conservative vector field의 문제점?
> ![[스크린샷 2025-07-20 오후 8.23.22.png]]
> 순수 $∇log\;p$ 라면 보장되던 SDE/ODE 샘플링 수렴성이 무너져, 일부 sample에 과도하게 몰리거나(mode collapse), 편향된 샘플이 나올 수 있다. 또한, 소용돌이 치는 방향성 때문에 가끔 샘플이 비현실적으로 튀는 문제가 발생할 수도 있다. 

그럼에도 불구하고, **Classifier-free Guidance**가 실험적으로는 매우 선명하면서도 Condition에 충실한 이미지를 뽑아냈다고 한다. <br>즉, <b><font color="#de7802">수학적 엄밀성을 조금 포기해도, 실제 생성 품질·효율 면에서는 “좋은 근사(non-conservative)”가 더 강력하게 작용할 수 있다</font></b>고 저자는 말하였다. <br><br><br>




### References 
1. https://ffighting.net/deep-learning-paper-review/diffusion-model/classifier-guidance/
2. https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/cfdg/
3. https://www.youtube.com/watch?v=E0ksC0fTN_Q&pp=0gcJCfwAo7VqN5tD