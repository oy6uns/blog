---
tags:
  - "#GNN"
  - "#GCN"
date: 2025-04-23
created: 2025-04-23
modified: 2025-07-08
---
> Semi-Supervised Classification with Graph Convolutional Networks (ICLR ‘17)
> https://arxiv.org/pdf/1609.02907

## 수식 (1)
$$ 
\mathcal{L}=\mathcal{L}_0+λ\mathcal{L}_{reg},\\ \; \\where\;\mathcal{L}_{reg}=∑_{i,j}A_{ij}∥f(X_i)−f(X_j)∥^ 2=f(X)^⊤Δf(X). 
$$

Node Classification task를 해결하기 위한 graph-based semi-supervised learning에서

<b><font color="#00b050">graph Laplacian regularization term을 loss함수에 추가함</font></b>으로써 정보가 그래프를 통해 smoothing될 수 있게 해준다.

기존의 $\mathcal{L}_0$는 labeled part에 대한 supervised loss를 나타낸다.

### 그럼 $\mathcal{L}_{reg}$는 어떻게 유도될까?

그래프 정규화 항은 <b><font color="#00b050">인접한 노드들의 특징 벡터 차이를 최소화하는 역할</font></b>을 한다!

$$
\begin{array}{rl}
\mathcal{L}_{reg} = & \sum_{i,j} A_{ij}(f(X_i)^\top f(X_i) - 2f(X_i)^\top f(X_j) + f(X_j)^\top f(X_j)) \\
= & 2(f(X)^\top D f(X) - f(X)^\top A f(X)) \\
= & 2(f(X)^\top (D - A) f(X)) \\
= & 2(f(X)^\top \Delta f(X)) \\
\end{array}
$$

여기서,

- $A$ : 그래프의 **인접 행렬 (Adjacency Matrix)**
- $D$ : 그래프의 **차수 행렬 (Degree Matrix)** , 즉 $D_{ii} = ∑_j A_{ij}$
- $Δ = D−A$ : **그래프 라플라시안 행렬 (Graph Laplacian)**

> [!missing] 문제점
> 기존의 방식은 수식 (1)과 같이 **$\mathcal{L}$(loss)에 추가적인 regularization 항을 추가하는 방식**을 통해 노드 간의 정보가 학습 시에 전달 될 수 있도록 하였다. 그러나, 학습하는 함수 $f(X)$ 자체에 노드 간의 연결성은 고려되지 않는다.

> [!success] 해결 방안
> <font color="#00b050">이 논문은 학습하는 함수 $f(X, A)$ 를 제시하여</font>
> 연결된 노드들의 표현을 유사하게 만들도록 강제하는 정규화 항을 손실함수에 추가하는 대신, <font color="#00b050">신경망 자체가 그래프 구조(인접행렬 $A$)를 직접 학습하도록 설계한다.</font>
> 
> → 인접행렬을 학습시에 사용하면, <font color="#00b050">⭐️ 레이블이 있는 노드뿐만 아니라 없는 노드의 정보도 학습할 수 있게 된다.⭐️</font>

## 논문의 2가지 Contribution
1. graph에서 **직접적으로 작동하는 neural networks model**을 설계하기 위한 simple & well behaved **layer-wise propagation rule 을 제시**한다. 
   그리고 위 rule이 <b><font color="#f79646">어떻게 spectral graph convolution의 1차 근사치로부터 유도될 수 있는지</font></b>를 보인다.
2. 위의 모델이 <b><font color="#f79646">fast & scalable semi-supervised classification of nodes in a graph</font></b>를 잘 달성한다는 것을 보인다.

## 수식 (2)
는 최종식이라, 나중에 모든 정리가 끝나고 다시 다루기로 하자.


# 2.1 Spectral Graph Convolutions

## 💡 Contribution #1.
<b><font color="#0070c0">------------</font></b>
## 수식 (3) = 수식 (7) 을 유도하는 것이 핵심이다.

$$
g_θ \;⋆\;x=Ug_θU^⊤x \;\;\cdot\cdot\cdot\;\;(3) 
$$

$$
g_θ \;⋆ \;x ≈ θ(I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x\;\;\cdot\cdot\cdot\;\; (7)
$$

수식 (3)은 filter를 통해 spectral convolution을 정의하는 것이고, 수식 (7)은 결과적으로 convolution 식이 인접 행렬(자기자신) 및 라플라시안 행렬(주변 노드)의 합으로 유도될 수 있다는 것을 보여준다. 이제 하나씩 알아보자.

## 수식 (3)
Graph Signal $x \in R^N$와 filter $g_\theta$ 에 대해 spectral convolution은 다음과 같이 정의된다.

$$
g_θ \;⋆\;x=Ug_\theta(Λ)U^⊤x \;\;\cdot\cdot\cdot\;\; (3)
$$

1. $x$ : 각 노드가 가지고 있는 특징(feature) 값
2. $U$: 정규화된 그래프 라플라시안 $L$의 고유벡터 행렬
3. $Λ$ : 정규화된 그래프 라플라시안의 고유값을 포함하는 대각행렬(고유값 행렬)
    - 이 값들은 <b><font color="#f79646">그래프의 스펙트럼 정보를 담고 있음.</font></b>
4. $g_\theta(Λ)$: 고유값 $Λ$를 이용해 그래프의 주파수를 조정하는 필터
	- 그래프의 <b><font color="#f79646">특정 주파수 대역을 강조하거나 억제하는 역할</font></b>을 한다.

$$ 
g_{\theta}(\Lambda) =\begin{bmatrix}g_{\theta}(\lambda_0) & & \\& g_{\theta}(\lambda_1) & \\& & \ddots & \\& & & g_{\theta}(\lambda_{N-1})\end{bmatrix}
$$


5. $Ug_θU^⊤$: 필터를 적용

> [!success] 전체 과정을 정리하면, **graph convolution** $g_θ \;⋆\;x$ 는
> 1. 그래프의 신호를 **푸리에 변환하여 : $U^Tx$**
> 2. **주파수 공간에서 $g_\theta$를 통해 조정**한 후 : $g_\theta(Λ)U^Tx$
> 3. **다시 원래 공간으로 변환**하는 것이다! : $Ug_\theta(Λ)U^Tx$

수식 (3)의 우변은 정규화된 그래프 라플라시안 L을 고유값 분해한 것이다.

> 참조: [https://deep-learning-study.tistory.com/414](https://deep-learning-study.tistory.com/414)

💡 그래프 라플라시안은 “real symmetric matrix” 이기 때문에 항상 고유값 분해(eigenvalue decomposition)이 가능하다.
따라서, $L = I-D^{-1/2}AD^{-1/2}$의 eigenvector로 이루어진 Fourier basis이고, $L = UΛU^T$로 표현할 수 있다.

### 그러나 이 방법은 비효율적이다!
1. 계산 복잡도가 너무 큼
    고유벡터 행렬 $U$는 $N \times N$ 크기의 행렬이므로, 이를 이용한 연산은 $O(N^2)$의 복잡도를 가진다.    
2. 고유 분해(Eigendecomposition) 연산이 비싸다
    - $L$의 고유분해를 수행해야 $U$와 $Λ$를 얻을 수 있다.
    - 이 연산은 $O(N^3)$의 복잡도를 가진다 → 매우 느리다!
→ 따라서, 이 방식은 이론적으로는 문제가 없지만, **node의 개수가 수천 수만개인 그래프에 대해서 위를 계산하는 것**은 **현실적으로 너무 비효율적**이다.

이를 해결하기 위해, truncated Chebyshev expansion을 통해 $g_\theta(Λ)$를 다항식으로 근사한다.
## 수식 (4)

$$
g_{θ^′}(Λ) ≈ \sum^K_{k=0} \theta'_kT_k(\tildeΛ)
$$

여기서 $\tildeΛ = \frac{2}{\lambda_{max}}Λ-I$로 정의한다.

Chebyshev 다항식은 $(−1,1)$에서 동작하므로, $L$의 가장 큰 eigenvalue $\lambda_{max}$를 사용해 $Λ$를 scaling$(0, \lambda_{max})\rightarrow (-1, 1)$ 해준 것이다.

수식 (4)의 근사를 수식 (3)에 대입하면, <br>

$\tilde L = \frac{2}{\lambda_{max}}L-I$에 대해,
## 수식(5)

$$
g_θ \;⋆\;x≈\sum^K_{k=0} \theta'_kT_k(\tilde L)x = y
$$

위 수식이 특별한 이유는 각 node에 대해 localized 되어 있기 때문이다. 우선 graph Laplacian $L$은 다음과 같이 localization 특성을 가진다.

> $(L^s)_{ij}$_는 그래프의 두 node $i$와_ $j$_를 연결하는 path 들 중 
> 길이가_ $s$_이하인 path 들의 개수와 일치한다._

수식 (5)에서 $L$의 $K$th power까지만 존재하기 때문에, $y(i)$는 $i$의 $K$-th order neighborhood signal 들의 합으로 표현할 수 있다!

✨ **따라서 수식 (5)의 근사는 $K$-localized 됨을 확인할 수 있다!!!**

# 2.2 Layer-Wise Linear Model

수식 (5)에서 $K$가 클수록 더 많은 종류의 convolutional filter를 얻을 수 있다.

그러나 그만큼 계산이 복잡해지며, overfitting의 가능성도 커진다.

🖍️ **여러 개의 convolutional layer를 쌓아 deep model**을 만든다면, $K$가 작아도 다양한 종류의 convolutional filter를 표현할 수 있다. 특히 **overfitting의 가능성을 덜 수 있고, 한정된 자원에 대해서 $K$가 클 때보다 더 깊은 모델을 만들 수 있다.**

이 논문에서는 극단적으로 $K=1$로 제한을 두었다.

또한 normalized graph Laplacian$\big(L = I_N − D^{-\frac{1}{2}}AD^{-\frac{1}{2}}\big)$의 eigenvalue들은 $[0, 2]$ 구간에 속하기 때문에,

> [!callout]- 참고
> ![[Pasted image 20250424005022.png]]

$\lambda_{max}\approx2$ 로 근사한다. 따라서, $\tilde L = \frac{2}{\lambda_{max}}L-I = L-I = -D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$가 된다.

> [!callout]- 참고
> ### 정규화된 그래프 라플라시안?
> **정규화된 그래프 라플라시안**은 다음과 같이 정의된다.
> $$ L_{\text{norm}} = I_N - D^{-1/2} A D^{-1/2} $$
> 여기서,
> - **$I_N$**: 자기 자신을 의미하는 단위 행렬
> - **$D^{-1/2} A D^{-1/2}$**: 차수에 따라 정규화된 이웃 정보
> 이 식을 다시 보면:
> $$ L_{\text{norm}} X = (I_N - D^{-1/2} A D^{-1/2}) X $$
> **정규화된 인접 행렬**은 아래와 같이 나타낼 수 있다.
> $$ 
> D^{-1/2}AD^{-1/2} = \begin{bmatrix} 0 & \frac{1}{\sqrt{d_1d_2}} & \frac{1}{\sqrt{d_1d_3}} & \frac{1}{\sqrt{d_1d_4}} \\ \frac{1}{\sqrt{d_1d_2}} & 0 & \frac{1}{\sqrt{d_2d_3}} & 0 \\ \frac{1}{\sqrt{d_1d_3}} & \frac{1}{\sqrt{d_2d_3}} & 0 & 0 \\ \frac{1}{\sqrt{d_1d_4}} & 0 & 0 & 0 \end{bmatrix}  \\ 
> $$ 
> $$ 
> I_N - D^{-1/2}AD^{-1/2} = \begin{bmatrix} 1 & -\frac{1}{\sqrt{d_1d_2}} & -\frac{1}{\sqrt{d_1d_3}} & -\frac{1}{\sqrt{d_1d_4}} \\ -\frac{1}{\sqrt{d_1d_2}} & 1 & -\frac{1}{\sqrt{d_2d_3}} & 0 \\ -\frac{1}{\sqrt{d_1d_3}} & -\frac{1}{\sqrt{d_2d_3}} & 1 & 0 \\ -\frac{1}{\sqrt{d_1d_4}} & 0 & 0 & 1 \end{bmatrix}
> $$
> > 즉, **자기 자신**($I_N X$)에서 **정규화된 이웃 정보**($D^{-1/2} A D^{-1/2} X$)를 빼는 연산을 수행.
> - **그래프에서 노드가 주변과 얼마나 다른지를 분석**할 수 있다.
> - 결과적으로 **노드 간 차이를 줄이고, 스무딩** 역할을 수행한다.

$T_0(x) =1, T_1(x) =x$ 이므로, 대입해서 정리한 식은 다음과 같다.
## 수식 (6)

$$
g_θ \;⋆\;x≈\sum^K_{k=0} \theta'_kT_k(\tilde L)x = \theta'_0x+\theta'_1(L-I)x = \theta'_0x-\theta'_1D^{-\frac{1}{2}}AD^{-\frac{1}{2}}x
$$

더 나아가, 계산을 줄이기 위해 $parameter\; \theta = \theta'_0 = -\theta'_1$ 만을 사용한다면, 다음과 같은 간단한 결과를 얻을 수 있다!
## 수식 (7)

$$
g_θ \;⋆\;x≈\theta \Big(I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}} \Big)x
$$

$D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$의 eigenvalue는 $[-1, 1]$ 구간에 속하기 때문에,

$I_N +D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$의 eigenvalue는 $[0, 2]$ 구간에 속한다.

만약 수식 (7)의 layer를 여러개 쌓아 deep model을 만든다면, eigenvalue가 $[0, 1]$ 범위 안에 들어오지 않기에 exploding / vanishing gradient problem이 생겨 불안정한 학습이 이루어진다!

## 💡 Contribution #2.
<b><font color="#0070c0">------------</font></b>

따라서 논문에서는 이를 해결하기 위해 renormalization trick 을 사용한다!!

**Renormalization Trick**을 적용하면 인접 행렬을 다음과 같이 바꾼다.

- 자기 자신 연결(Self-loop) 추가:  
$$ 
\tilde{A} = A + I 
$$
- 새로운 차수 행렬:  
$$ 
\tilde{D}_{ii} = \sum_j \tilde{A}_{ij} 
$$
- 새롭게 정규화된 인접 행렬:  
$$ 
\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} 
$$
새롭게 정규화된 인접행렬의 eigenvalue는 $[0, 1]$ 범위로 좁혀진다!!!

이렇게 되면 고유값이 더 안정적인 범위에 놓이고, 모델이 여러층을 쌓아도 기울기 폭발/소실 문제가 줄어들게 된다!!

**결과적으로 $I_N +D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ 대신, $\tilde D^{-\frac{1}{2}}\tilde A\tilde D^{-\frac{1}{2}}$ 를 이용해 다음과 같이 convolutional filter를 정의한다.**

$$
g_θ∗x≈θ\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}x
$$

위의 결과는 각 node가 1차원의 feature를 가질 때로 한정되어 있다. 이제 각 node마다 $C$차원의 feature vector를 가지는 상황을 고려해보자!

- 입력 신호 $X \in R^{N\times C}$
    총 $N$개의 노드(feature)에 대해 각 노드(feature vector)가 $C$-차원을 가질 때, 입력 신호 $X$는 $X \in R^{N\times C}$가 된다.
- 필터 행렬 $Θ \in R^{C\times F}$
    $F$개의 필터를 가진다고 할 때, **입력 차원 $C$를 출력 차원 $F$로 변환**시켜준다.
- 출력 신호 $Z \in R^{N \times F}$
    - 입력 $X$에 대해 $F$개의 필터를 적용하여 새로운 특징을 추출한다.
    - $N$개의 노드에 대해 $F$개의 feature를 가지게 된다!

결과적으로 일반화된 식은 아래와 같다.
## 수식 (8)
$$
Z = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}XΘ
$$

그래프의 엣지 개수를 $|\mathcal{E}|$라고 하면,
각 노드는 평균적으로 $\frac{|\mathcal{E}|}{N}$개의 이웃을 가지므로, 전체 노드 $N$개에 대해
**time complexity**는 $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}X$ 계산 시에 $\mathcal{O(|\mathcal{E}|}C)$이고,
$Θ \in R^{C\times F}$를 곱할 때, $F$만큼 추가되어 $\mathcal{O(|\mathcal{E}|}CF)$가 된다!

수식 (8)을 사용해 **multi-layer GCN의 layer-wise propagation rule 을 정의**할 수 있다.
$l$번째 layer와 $l+1$번째 layer의 activation을 다음과 같이 정의하면,
- $H^{(l)} \in \mathbb{R}^{N \times C_l}$
- $H^{(l+1)} \in \mathbb{R}^{N \times C_{l+1}}$
이때, 학습 가능한 가중치 행렬 $W^{(l)} \in \mathbb{R}^{C_l \times C_{l+1}}$ 와 활성화 함수 $\sigma$(예: ReLU, tanh)를 사용하여 layer-wise propagation rule을 정의하면:
$$
H^{(l+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)
$$
이 공식은 GCN에서 각 레이어의 특징을 다음 레이어로 전달하는 방법을 나타낸다!

# 3. Semi-Supervised Node Classification
![[Pasted image 20250423164351.png]]
GCN Paper에서는 2-Layer GCN을 예로 들어 설명한다.
## 수식(9)

$$
Z=f(X,A)=softmax(A^⋅ReLU(A^XW^{(0)})W^{(1)})
$$

위와 같이 두 개의 layer를 가지는 model을 만들 수 있다.

### **여기서 각 기호의 의미는 다음과 같다:**
- **$X \in \mathbb{R}^{N \times C}$**: 노드 특징 행렬 (각 노드는 C-차원의 입력 feature를 가짐)
- **A** : 인접 행렬 (노드 간 연결 정보를 나타냄)
- **$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$** : 정규화된 인접 행렬 (**$\tilde{A}=A+I$** 는 자기 자신 연결(self-loop)이 포함된 인접 행렬)
- **$W^{(0)} \in \mathbb{R}^{C \times H}$** : 입력층에서 은닉층으로 가는 가중치 행렬
- **$W^{(1)} \in \mathbb{R}^{H \times F}$** : 은닉층에서 출력층으로 가는 가중치 행렬
- **ReLU** : 활성화 함수
- **softmax** : 최종 출력층에서 분류 확률을 계산하는 함수

## 수식(10)

$$
\mathcal{L} = - \sum_{l \in \text{labeled}} \sum_{f=1}^{\text{output dim}} Y_{lf} \ln Z_{lf}
$$

마지막 output layer를 거친 후,
Loss function으로 **label 이 있는 node 들에 대해서만 cross-entropy error 를 계산**한다.
이를 통해 수식 (9)의 weight matrix $W^{(0)}$와 $W^{(1)}$은 gradient descent를 통해 업데이트 한다!

# 5. Experiments
크게 네 가지 dataset: Citeseer, Cora, Pubmed, NELL 을 실험에 사용하였다.
각 데이터셋에 대한 **baseline method** 들과 **two-layer GCN 의 classification accuracy** 는 다음과 같다.
![[Pasted image 20250423164504.png]]
GCN 의 정확도가 다른 baseline method 들에 비해 월등히 높은 것을 볼 수 있다. 
특히 baseline method 들 중 <font color="#00b050">정확도가 가장 높은 Planetoid 와 비교해, GCN 의 수렴 속도가 훨씬 빠르다</font>는 것을 알 수 있다.

![[Pasted image 20250423164527.png]]
✅ 수식 (8)에서 사용한 <font color="#00b050">renormalization trick이 가장 높은 정확도</font>를 보여주는 것을 확인할 수 있다!
