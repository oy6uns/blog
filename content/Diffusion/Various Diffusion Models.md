---
date: 2025-07-18
created: 2025-07-18
modified: 2025-07-18
tags:
---
[[DDPM]] 내용을 안다는 가정 하에 아래 다양한 Diffusion 논문들은 핵심들만 짚어서 정리하였다. <br><br>
# Classifier Guidance 
> Diffusion Models Beat GANs on Image Synthesis (NeurIPS workshop ‘21)<br>https://arxiv.org/pdf/2105.05233

다양한 Diffusion Model들이 발표되었지만, GAN과 비교했을 때 이들 방법의 공통적인 한계는 **Condition을 반영할 수 없다는 것**이었다. <br>Classifier Guidance에서는 Class 정보를 Condition으로 입력 받아 <b><font color="#e36c09">해당 Class 이미지만 생성할 수 있는 Diffusion Model을 제안</font></b>한다. **별도의 Classifier를 학습시켜 이로부터 나오는 Gradient를 활용**하기 때문에 Classifier Guidance라는 별칭이 붙었다. <br><br>두 가지 Contribution이 있다. 
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
## Classifier Guidance
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
분산 $\Sigma$는 그대로 두고, 평균만 이동시킨 뒤<br>$\mathcal{N}(\tilde{\mu}, \Sigma)$에서 샘플을 뽑으면, <br>**”클래스 $y$ 쪽으로 가중된” Conditional Reverse Process**가 된다!


# Classifier-free Guidance
> Classifier-Free Diffusion Guidance (ICLR’ 22)<br>https://arxiv.org/pdf/2207.12598

Classifier를 추가적으로 학습해야 한다는 단점을 극복하기 위해 나온 논문이다. <br><br>Classifier Guidance는 
1. 추가 Classifier를 학습해야 하고, 
2. 그 Classifier는 노이즈가 섞인 이미지로 학습해야 하므로 Pretrained된 일반적인 Classifier를 바로 사용할 수 없으며, 
3. 확산 모델의 본래 스코어(“데이터 분포에서 자연스러운 이미지 방향”)와 Classifier Gradient(“분류기에는 $y$로 보이는 방향”)는 항상 일치하지 않는다. 따라서, 정도를 잘못 조절하면 모델이 학습한 **“자연스러운 분포”** 에서 꽤 벗어난 지점까지 샘플이 이동하게 될 수 있다. <br>
는 단점들이 존재한다. <br>이 논문의 핵심은 **Classifier 없이 Diffusion 모델을 Guidance**할 수 있는 방법을 제시한다. 따라서 **“Classifier-free Guidance”** 라는 이름으로 더 널리 알려져있다. 

본 논문은 단순히 샘플이 품질을 SOTA로 올리는 것에 집중하기 보다, **classifier-free guidance가 classifier guidance와 비슷하게 FID/ID를 달성할 수 있는지**를 보이고, classifier-free guidance를 이해하는 것을 목표로 한다. 

















### References 
1. https://ffighting.net/deep-learning-paper-review/diffusion-model/classifier-guidance/
2. https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/cfdg/
3. 