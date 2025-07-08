---
tags:
  - GNN
  - Dynamic_GNN
  - Recsys
  - Sequential
---
> Dynamic Graph Neural Networks for Sequential Recommendation (TKDE ‘22)
> https://arxiv.org/pdf/1609.02907

# 1. Introduction
## 기존 추천시스템
- 예전 추천시스템(Collaborate Filtering)은 주로 **정적인 사용자-아이템 관계**만 본다. 
	- 즉, “A가 B를 샀으니 비슷한 C도 살 거다” 처럼 한 시점의 “취향”만 반영
- 현실에서는 사용자의 **취향이 시계열적으로(순차적으로) 변함**. 예를 들어, 영화/음악/상품 취향이 최근에 본 것(들은 것, 산 것)에 영향을 받음.
- 그래서 최근에는 사용자의 **Sequential Information을 반영하는 연구가 많아졌다**. 
	- Markov chain, RNN/LSTM/GRU, CNN 및 Attention 기반 연구

## 기존 접근법의 공통적인 한계점
- **단일 사용자의 시퀀스만**(A의 구매, 클릭, 시청 기록) 집중적으로 보고, **”다른 사용자들과의 동적 협업 신호(동시적, 혹은 유사한 시점의 연관된 행동들)"** 는 잘 고려하지 못함.

## 제안한 방법
- **모든 사용자 시퀀스**를 **동적 그래프**에 통합(엣지에 타임스탬프/순서 정보 부여).
- **Dynamic Graph Recommendation Network(DGRN)** 라는 새로운 GNN 구조를 설계해, 이 서브그래프에서 사용자의 장-단기 선호를 추출.
- **다음 아이템 예측**은 '동적 그래프 내에서 사용자 노드와 아이템 노드가 새로 연결될 확률'을 구하는 **링크 예측 문제**로 변환.

# 2. Related Work
## 2.1 Sequential Recommendation
#### Markov chain
- [Factorizing Personalized Markov Chains for Next-Basket Recommendation](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p811.pdf)
- [Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation](https://arxiv.org/pdf/1609.09152) ✅
#### Translation-based approaches
- [Translation-based recommendation](https://arxiv.org/pdf/1707.02410)
### Deep-learning based
#### RNN
- [GRU4Rec](https://arxiv.org/pdf/1511.06939) - firsst one to use RNN to session-based recommendation task
- RNN은 sequential recommendation task에 효과적이라는게 입증되어 widely 사용되었다.
#### CNN
- sequential data에서의 다른 패턴을 찾아내기 위해 CNN도 사용되었다. [Caser](https://arxiv.org/pdf/1809.07426)
#### Attention Mechanisms
- item들 간의 relationship을 포착하기 위해 attention도 효과적으로 사용되었다. 
#### GNN 
최근 몇 년간의 연구를 통해, 그래프 구조 데이터를 처리하는 데 있어 그래프 신경망(GNN)이 가장 효과적인 방법 중 하나로 밝혀졌으며, 다양한 작업에서 SOTA(State-of-the-Art) 성능을 달성하였다.
- [SR-GNN](https://arxiv.org/pdf/1811.00855) - Gated GNNs
- [A-PGNN](https://arxiv.org/pdf/1910.08887) - combining personalized GNN & attention mechanism 
- [MA-GNN](https://arxiv.org/pdf/1912.11730) - augmented graph neural network to **capture both long–term & short-term user interests**

> [!warning] GNN 기반 sequential recommendation의 한계
> - 최근의 GNN(그래프 신경망) 기반 순차 추천 모델들은 **사용자 자신 시퀀스 내부(내부 시퀀스, intra-sequence)** 정보만을 바탕으로 사용자의 선호를 학습한다. 
> - 즉, **여러 사용자 시퀀스(다른 사용자들의 행동 흐름) 사이에서 아이템이 어떻게 연결·공유되고 있는가** 하는 **시퀀스 간(item relationship across sequences)** 연관성은 제대로 반영하지 못한다. 

이러한 한계를 보완하기 위해 여러 모델들이 제안되었다. 
- [HyperRec](https://people.engr.tamu.edu/caverlee/pubs/wang20next.pdf) - Hypergraph라는 구조를 활용하여 둘 이상의 아이템을 한 번에 엣지로 묶는다. → 이를 이용해 sequence 안팎의 아이템 간 high-order 상관관계를 모델링한다. 
	- 실제 상호작용의 세밀한 시간, 순서 정보는 활용하기 어렵다. 
- [CSRM](https://dl.acm.org/doi/pdf/10.1145/3331184.3331210) - neighborhood session similarity를 계산해 단순히 자기 세션만 보는 것이 아니라 주변 세션 정보도 활용한다. 
	- 세션 간 단순 유사도로 이웃 세션 정보를 연계하기 때문에 상호작용 구조나 시계열성은 놓칠 수 있다. 
- [DGRec](https://arxiv.org/pdf/1902.09362) - 소셜 관계가 있는 경우, 사용자 시퀀스끼리 직접적으로 연결하여 정보를 교환한다. 
	- 데이터에 소셜정보가 반드시 있어야만 추가 연결정보를 이용할 수 있다. 


> [!Important] DGSR: 모든 상호작용을 ‘동적 그래프’로 통합
> 사용자의 모든 행동 이력을 시간/순서 정보까지 포함하여, **"유저-아이템" bipartite graph**의 형태에 **동적 변화**(타임스탬프, 인터랙션 순서 등)까지 같이 녹여내었다.<br> 
> → 아무 추가 정보 없이도 동적 그래프 프레임워크 하나로 통합되어 **보조정보에 의존하지 않아도 된다는 장점**이 있다. 

## 2.2 Dynamic Graph Neural Networks
real-world 그래프(such as academic network, social network, and recommender system)는 **시간에 따라 node와 edge가 변화함**에 따라 동적으로 이를 처리해주어야 한다.

여러 Dynamic Graph 방법론들은 e-commerse dataset에서 테스트 되었지만, sequential recommendation 시나리오에서는 아직 적용되지 않았다. 

이 논문에서는 <b><font color="#00b050">dynamic graph의 관점에서 sequential recommendation 문제를 해결</font></b>하고자 한다. 

# 3. Prerequisite
## 3.1 Sequential Recommendation
사용자의 행동 기록을 기반으로 다음에 어떤 아이템을 선택할지를 에측하는 문제
#### 기호 설명
- $\mathcal{U}$: 사용자 집합
- $\mathcal{I}$: 아이템 집합
- 사용자 $u \in \mathcal{U}$의 행동 시퀀스 $S^u = (i_1, i_2, \dots, i_k)$): 시간 순서대로 본 아이템들
- $T^u = (t_1, t_2, \dots, t_k)$: 각 아이템을 본 시간
- 전체 시퀀스 집합: $\mathcal{S}$
#### 목적
주어진 시점 $t_k$까지의 행동을 바탕으로 다음에 사용자 $u$가 볼 아이템 $i_{k+1}$를 예측한다. 
#### 길이 제한
실용성을 위해 시퀀스의 최대 길이 $n$을 정하고, 가장 최근 $n$개의 아이템만 사용합니다.
#### 임베딩
- 사용자와 아이템은 각각 $\mathbb{R}^d$의 벡터로 임베딩됨
- 사용자 임베딩 행렬: $\mathbf{E}_U \in \mathbb{R}^{|\mathcal{U}| \times d}$
- 아이템 임베딩 행렬: $\mathbf{E}_I \in \mathbb{R}^{|\mathcal{I}| \times d}$

## 3.2 Dynamic Graph
시간에 따라 변화하는 그래프로, 두 가지 유형이 있다. 
- 이산 시간(dynamic discrete-time)
- 연속 시간(dynamic continuous-time) → 이 논문에서는 이쪽에 초점
#### 정의
그래프 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{T})$
- $\mathcal{V}$: 노드 집합 ($v_1, v_2, \dots, v_n$​) 
- $\mathcal{E}$: 시간에 따른 엣지 집합
- $\mathcal{T}$: 시간 집합
#### 엣지 
시간 $t$에 노드 $v_i$와 $v_j$​ 간의 상호작용을 의미하는 삼중항 $(v_i, v_j, t)$
#### 기능
엣지의 시간 순서를 기록하여 노드 간 관계의 변화를 추적
#### 임베딩
각 노드를 $\mathbb{R}^d$ 벡터로 매핑하는 함수 $f: \mathcal{V} \to \mathbb{R}^d$

# 4. Methodology
## DGSR model
> four components in the architecture

![[스크린샷 2025-05-21 오후 4.36.24.png]]
1. Dynamic Graph Construction 
2. Sub-graph Sampling 
	- 전체 그래프를 매번 처리하는 건 비효율적이므로, **관심 있는 사용자 중심의 로컬 서브그래프를 구성**하여 효율적이고 집중적인 학습이 가능케 한다. 
3. Dynamic Graph Recommendation Networks
	- 서브 그래프 안에서의 노드들 간 Message Passing과 Node Embedding 업데이트를 수행한다. 
4. Prediction Layer
	- User Node와 모든 Item Node 간의 연결 가능성(link prediction)을 계산하여, 가장 높은 점수를 가진 아이템을 추천한다. 
## 4.1 Dynamic Graph Construction
- **user $u$ - item $i$ bipartite graph**(이중 그래프)
- **사용자가 아이템과 상호작용**(예: 상품 클릭/구매/평가)하는 순간 
  → 해당 시점의 **유저-아이템 노드 사이에 edge가 생성**된다. 
- 각 edge는 $(u, i, t, o_{ui}, o_{iu})$의 형태의 정보를 가진다. 
	- 사용자 $u$
	- 아이템 $i$
	- 타임스탬프(t): 상호작용 발생 시간
	- $o_{ui}$: **사용자가 해당 아이템을 몇 번째로 상호작용**했는지(순서)
	- $o_{iu}$: **그 아이템의 입장에서, 전체 상호작용 사용자 중 몇 번째 사용자**인지(순서)

> [!check] 목적
> - 이렇게 구성한 동적 그래프는 순서, 시간 정보뿐 아니라 **다른 사용자와 같은 아이템을 매개로 한 간접 연결, High-Order 연관성**까지 한 번에 포착 가능하다. 
> - 기존의 정적 그래프(혹은 단순 sequence)에 비해:
> 	- “**누가–언제–어떤 순서로–어떤 아이템과” 상호작용**했는지
> 	- **시퀀스 간 연결망의 순간순간 변화를 모두 반영**

## 4.2 Sub-graph Sampling 
### 핵심 목표
- 동적 그래프는 매우 크고 복잡할 수 있으므로, <font color="#00b050"><b><font color="#00b050">각 사용자의 추천(예측) 시점마다 필요한 부분만 뽑아서</font></b></font> 작은 그래프(subgraph)를 구성한다. 
- 이렇게 해야 
	- 계산 효율성을 높이고, 
	- 불필요한 정보를 줄이며, 각 target user node에 의미 있는 동적 맥락을 전달할 수 있다. 

### 구체적인 샘플링 절차
1. **Anchor Node(기준 노드) 선택**
2. **1차 이웃(First-Order Neighbors) 샘플**
	- <b><font color="#00b050">사용자가 최근 상호작용한 n개의 아이템</font></b>들을 먼저 뽑음
		(n은 sequence 최대 길이, 예: 최근 구매 10개 등)
3. **2차 이웃 이후 확장**
	- 각 아이템에 대해, **해당 아이템을 상호작용한 다른 사용자들(1-hop neighborhood)도 추가로 탐색**
	- multi-hop(다단계) 탐색을 통해 그래프를 확장해나간다.

<br>→ 중복을 방지하기 위해 **이미 anchor node로 사용한 사용자, 아이템들은 샘플링에서 제외**시킨다. 

> [!note] sub-graph generation
> - 중복을 방지하기 위해 **이미 anchor node로 사용한 사용자, 아이템들은 샘플링에서 제외**시킨다. 
> - 이렇게 선택된 사용자, 아이템 노드와 그들 간의 상호작용 Edge로 구성된 작은(sub-graph) 그래프가 추려질 수 있다!
> 
> 결과적으로 **타겟 사용자의 시퀀스+주변 사용자의 맥락+동적(시간/순서 유지) 정보**가 유지된 "현실적인 컨텍스트 그래프"가 만들어지고, 이 그래프에서만 정보를 추출, 연산함으로써, **추천 성능과 효율을 동시에 잡을 수 있다!**

## 4.3 Dynamic Graph Recommendation Networks
대부분의 GNN과 유사하게, 다음 두 가지 주요 구성 요소로 이루어져 있다:
1. Message Propagation
2. Node Updating

## #1. Message Propagation
메시지 전파 메커니즘은 <font color="#00b050">다음 두 가지 방향으로 정보를 전달</font>한다. 
#### 1) 항목에서 사용자로의 전파 
- 사용자의 이웃은 사용자가 상호작용한 항목들이다. 
- 각 사용자 노드에서 두 가지 정보를 추출한다:
	- **장기 선호도(Long-term preference)**: 사용자의 고유한 특성과 일반적인 선호도를 반영
	- **단기 선호도(Short-term preference)**: 사용자의 최근 관심사를 반영
#### 2) 사용자에서 항목으로의 전파
- 항목의 이웃은 그 항목과 상호작용한 사용자들이다. 
- 각 항목 노드에서 두 가지 특성을 추출한다:
	- **장기 특성(Long-term character)**: 항목의 일반적인 특성 반영
    - **단기 특성(Short-term character)**: 항목의 최신 속성 반영

**항목에서 사용자로의 전파**와 **사용자에서 항목으로의 전파** 둘 다 장기, 단기 정보를 동시에 인코딩한다. 
### 장기 정보 인코딩 
논문에서는 장기 정보와 단기 정보를 인코딩하기 위한 다양한 방법을 제시한다. 
1. **그래프 합성곱 신경망(GCN)**: 모든 이웃 노드 임베딩을 직접 집계
2. **순환 신경망(RNN)**: GRU 네트워크를 사용하여 시퀀스 의존성 모델링
3. **동적 그래프 어텐션 메커니즘(DAT)**: 그래프 어텐션과 상대적 순서 정보를 결합한 메커니즘
### 단기 정보 인코딩
1. **마지막 상호작용**: 마지막 항목/사용자 임베딩을 단기 정보로 사용
2. **어텐션 메커니즘**: 마지막 항목/사용자와 이전 항목/사용자 간의 어텐션을 계산

> [!NOTE] 장기 & 단기
> - **장기**: GCN/RNN + relative-order Attention으로 이웃 전체에서 표현 추출
> - **단기**: 마지막 interaction과 히스토리 간 어텐션으로 뽑아낸 표현 

## #2. Node Updating
각 노드의 새로운 표현을 만들기 위해 
1. 방금 계산한 <font color="#00b050">장기 임베딩</font> $h_u^L$,
2. <font color="#00b050">단기 임베딩</font> $h_u^S$,
3. 그리고 **이전 레이어에서의 임베딩** h_u^{(l-1)}
이 세 가지를 **쭉 이어붙인(concatenate)** 뒤에, **선형 변환 + 활성화 함수(tanh)** 를 적용한다. 

$$
h_u^{(l)}​=tanh(W_3^{(l)}​[h_u^L​∥h_u^S​∥h_u^{(l−1)}​])
$$
$$
W_3^{(l)} \in \mathbb{R}^{d\times3d}
$$


