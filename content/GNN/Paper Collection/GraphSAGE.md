---
tags:
  - GNN
  - GraphSAGE
date: 2025-05-07
created: 2025-05-07
modified: 2025-05-07
---



> Inductive Representation Learning on Large Graphs (NeurIPS ‘17) <br>
> https://arxiv.org/pdf/1706.02216

> [!Abstract] 
> **GraphSAGE**는 그래프에서 본 적 없는 **새로운 노드도 주어진 이웃 노드들의 정보를 활용해 임베딩을 즉석에서 만들 수 있게 해주는(즉, inductive하게 해주는)** 방법을 제안한다. 

# 1. Introduction
- **기존 임베딩 방법**들은 거의 모두 **정해진(고정된) 하나의 그래프**에서 **훈련에 쓰인 노드만** 벡터로 바꿔준다(=transductive).
- 현실에서는 **새로운 노드, 새로운 그래프(새로운 논문, 유저, 게시글 등)** 들이 계속 등장하는데, 이런 경우 **무조건 전체를 다시 훈련하거나 추가 학습을 해야 하므로 비효율적**이다. 
- <font color="#e36c09">새로운 노드(예: Reddit의 새 포스트, YouTube의 새 영상)에 대해서 <b>빠르게 임베딩을 만들어내는 inductive 방식</b>이 꼭 필요하다.</font>

## Inductive Node Embedding의 어려움
- **Inductive(귀납적) 노드 임베딩**이란, 훈련 때 보지 않은 노드(또는 완전히 새로운 그래프)라도 임베딩을 만들어내는 것을 의미한다. 
- 이게 어려운 이유는, <font color="#e36c09">새로운 노드가 기존 임베딩 공간에 <b>"정확히"</b> 들어맞아야 하기 때문</font>이다. 즉, 네트워크 구조/이웃 정보를 잘 활용해서 그래프 내에서의 역할과 위치를 파악해야한다. 

# 3. Proposed method: GraphSAGE
> [!check] **new framework**
> - **GraphSAGE(SAmple and aggreGatE)** 라는 새로운 inductive node embedding 프레임워크를 제안한다. 
> - 노드를 '개별적으로' 저장하는 대신, **이웃 노드들의 특성을 샘플링하고 집계(aggregate)해서 임베딩을 만드는 '함수' 자체를 학습**한다. 

- 특히, 이웃 노드 구조/특성을 이용해서 새로운 노드의 카테고리를 잘 예측하고, 완전히 새로운 그래프(단백질-단백질 네트워크)에서도 잘 동작함을 보인다. 

## 3.1 Embedding generation algorithm 
> [!Information] GraphSAGE embedding generation
> 1. **초기화**
> 	각 노드의 0단계 Embedding($h^0_v$)는 해당 노드의 Feature($x_v$)로 설정
> 2. **집계 및 임베딩 생성의 반복**
> 	- for $k=1$ to $K$:
> 		- a. 각 노드 $v$에 대해 이웃 $N(v)$의 $k-1$번째 $h^{k-1}_u$ 들을 집계(aggregate) →  $h^k_{N(v)}$   
> 		- b. 자신의 $k-1$ 번째 임베딩 $h^{k-1}_v$ 와 $h^k_{N(v)}$를 **concat**  
> 		- c. 이 벡터에 가중치 $W_k$, 비선형 $\sigma$을 적용해 $h^k_v$ 생성
> 	- $h^k_v$ 를 L2 정규화(벡터 길이 1로 만듦)
> 3. **최종 임베딩**
> 	$h^K_v$를 $z_v$로 반환

- **이웃 정보와 자체 정보를 반복적으로 K번 혼합**하여 K가 커질수록 멀리 있는 이웃의 정보까지 반영한다. 
- 모델이 학습된 후, 임의의(새로운) 노드 feature만 주면 이 <span style="background:rgba(255, 238, 131, 0.55)">"집계함수(Aggregation Function)"</span>와 <span style="background:rgba(255, 238, 131, 0.55)">"가중치(Weight)"</span>를 통해 임베딩을 빠르게 만들 수 있어 <span style="background:rgba(255, 238, 131, 0.55)">inductive (새 노드를 처리) 능력</span>이 생긴다. 


> [!important] Mini-Batch 학습
> **대형 그래프 전체를 한 번에 메모리에 올리거나, 모든 노드의 임베딩을 계산**한다면 메모리 부담이 커진다. 
> 따라서, GraphSAGE는 학습할 때 타겟 노드 중심으로 역순으로 이웃으로 샘플링하고, 샘플링된 이웃들의 정보를 **hop을 기준으로 Batch를 묶는다.** 
> 
> **K=2**기준,
> - $B_0$: **target node**(embedding을 알고 싶은 노드들)
> - $B_1$: target node들의 **1-hop 이웃 node들**
> - $B_2$: target node들의 **2-hop 이웃 node들** 
> 
> ✅ 최종적으로 K만큼의 집합이 생긴다. → mini-batch 처리가 가능해짐

### Weisfeiler-Lehman (WL) Isomorphism Test와의 비교
WL test는 **그래프 동형성(isomorphism, 구조가 같은지) 판단을 위한 오래된 알고리즘**이다. 
- <u>핵심 아이디어</u>
	- 각 노드의 “레이블”을 이웃 노드들의 레이블 정보를 집계하여 반복적으로 업데이트한다. 
	- 매 step마다 이웃들의 정보를 합쳐 “새로운 레이블(혹은 색깔)”을 부여, 이 과정을 여러 번 반복한다. 

만약, 
- $K=|V|$
- 각 레이어의 가중치 $W_k$를 항등 행렬로 고정
- Activation 없이
- Aggreagate 함수를 **“set의 모든 아이템을 받아, 합친 뒤 해시(hash)로 고유값을 내는 함수”** 로 두면

→ 이 업데이트는 WL 알고리즘에서의 **“새로운 레이블 만들기”** 와 동일해진다. 

> [!check] **즉,** GraphSAGE**는** WL-Test의 연속적(continuous), 미분가능, 신경망적 근사판**이다!**
> 미분 불가능한 **WL-Test의 해시(hash, discrete) 대신**에 **신경망 기반 합성함수와 non-linearity**를 사용해서 연속적인 벡터표현을 만들고, 학습도 가능하게 확장한 것!


GNN의 구별할 수 있는 그래프(pairwise) **WL Test 이하임은 후속 연구에서 증명**되었다. 
> (자세한 내용: [Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019][[https://arxiv.org/abs/1810.00826](https://alphaxiv.org/abs/1810.00826)])

## 3.2 Learning the parameters of GraphSAGE
$$
J_G​(z_u​)=−log(σ(z_u^⊤​z_v​))−Q⋅E_{{v_n​}∼{P_n​(v)}}​[log(σ(−z_u^⊤​z_{v_n}​​))]
$$
negative sampling 기반의 contrastive loss 식이다. 
![[스크린샷 2025-05-15 오후 8.56.00.png]]
즉, 이 손실 함수는 다음을 유도한다. 
- **양성 쌍(positive pair)** 인 ($z_u$, $z_v$)는 내적값이 크게 → 시그모이드 값이 1에 가깝게 학습
- **음성 쌍 (negative pair)** 인 ($z_u$, $z_v$)는 내적값이 **작게 (또는 음수로) → 시그모이드 값 0에 가깝게** 학습됨

#### 뒤의 항에서 기댓값을 쓰는 이유?
- 모든 노드를 부정 샘플로 넣기엔 너무 많으니까 → 일부만 샘플링
- 확률 분포 $P_n(v)$를 따라 샘플링하고,
- 그 평균 값을 계산함으로써 전체 분포의 효과를 근사

## 3.3 Aggregator Architectures
GraphSAGE가 노드 임베딩을 생성할 때 "이웃 노드의 feature를 어떻게 집계(aggregate)할 것인가"에 대한 다양한 **집계 함수(aggregator functions)** 를 제안한다. 이 집계 함수들은 **순서가 없는(unordered) 이웃**의 feature를 임베딩에 반영해야 하므로, 대칭적이고(순서 무관), trainable한 구조가 필요하다. 
### 주요 Aggregator 종류
1. Mean Aggregator
2. LSTM Aggregator 
	- 이웃에 대해 임의의 순서를 부여한 뒤 LSTM에 넣음
3. Pooling Aggregator 
	- 각각의 이웃 임베딩을 독립적으로 작은 MLP에 통과시키고, element-wise max pooling이나 mean pooling으로 집계. 
	- $AGGREGATE^{pool}_k = max({σ(W_{pool}h^k_{u_i} + b) , ∀u_i ∈ N(v)})$

> [!check] 
> 이 세 종류 모두 permutation-invariance(입력 순서와 무관성)를 일정 수준 가지고 있어, 임의 순서의 이웃 node 집합에도 robust함.

# 4. Experiments
![[스크린샷 2025-05-15 오후 9.36.06.png]]
- DeepWalk보다 Inference Time이 작은 것을 확인할 수 있다. 
- **각 노드의 이웃 전체**를 모두 집계(aggregation)하지 않고, **"고정된 개수"의 이웃 노드만 무작위로 샘플링**해서 사용한다. 이때, 몇 명의 이웃을 볼지가 **Neighborhood Sample Size**이다. 
- Neighborhood는 많이 볼수록 성능이 좋아지지만, runtime 또한 exponentially하게 증가하는 것을 볼 수 있음. 

## 4.2 Generalizing across graphs
기존의 DeepWalk 방법은
각 그래프마다 랜덤한 축에 단어 위치를 고정시켰다. 

GraphSAGE는 이 대신에 ==임베딩 생성 함수 자체를 학습==한다. 
따라서, 새로운 그래프에 적용하는 것이 가능하다. **Aggregate하는 함수만 재사용**하기 때문에 **모든 그래프/노드에 대해 feature + neighborhood → 임베딩**을 “동일한 방식”으로 생성할 수 있다. 
$$
h_v^{(k)}​=σ(W^{(k)}⋅CONCAT(h_v{(k−1)}​,AGGREGATE({h_u^{(k−1)}​∣u∈N(v)})))
$$
- $h_v^{(k)}$​: 노드 $v$의 $k$번째 레이어 임베딩
- $\mathcal{N}(v)$: 노드 $v$의 이웃들
- $W^{(k)}$: 학습 가능한 가중치
- $\sigma$: 비선형 함수 (e.g. ReLU)
