> Inductive Representation Learning on Large Graphs (NeurIPS ‘17)
> https://arxiv.org/pdf/1706.02216

> [!important] 
> **GraphSAGE**는 그래프에서 본 적 없는 새로운 노드도 주어진 <span style="background:#d3f8b6">이웃 노드들의 정보를 활용해 임베딩을 즉석에서 만들 수 있게 해주는</span>(즉, inductive하게 해주는) 방법을 제안한다. 

<br>
# 1. Introduction
- **기존 임베딩 방법**들은 거의 모두 **정해진(고정된) 하나의 그래프**에서 **훈련에 쓰인 노드만** 벡터로 바꿔준다(=transductive).
- 현실에서는 **새로운 노드, 새로운 그래프(새로운 논문, 유저, 게시글 등)** 들이 계속 등장하는데, 이런 경우 **무조건 전체를 다시 훈련하거나 추가 학습을 해야 하므로 비효율적**이다. 
- <span style="background:rgba(255, 238, 131, 0.55)">새로운 노드(예: Reddit의 새 포스트, YouTube의 새 영상)에 대해서 <b>빠르게 임베딩을 만들어내는 inductive 방식</b>이 꼭 필요하다.</span>

## Inductive Node Embedding의 어려움
- **Inductive(귀납적) 노드 임베딩**이란, 훈련 때 보지 않은 노드(또는 완전히 새로운 그래프)라도 임베딩을 만들어내는 것을 의미한다. 
- 이게 어려운 이유는, <span style="background:rgba(255, 238, 131, 0.55)">새로운 노드가 기존 임베딩 공간에 "정확히" 들어맞아야 하기 때문</span>이다. 즉, 네트워크 구조/이웃 정보를 잘 활용해서 그래프 내에서의 역할과 위치를 파악해야한다. 

## GraphSAGE
> [!check] **new framework**
> - **GraphSAGE(SAmple and aggreGatE)** 라는 새로운 inductive node embedding 프레임워크를 제안한다. 
> - 노드를 '개별적으로' 저장하는 대신, **이웃 노드들의 특성을 샘플링하고 집계(aggregate)해서 임베딩을 만드는 '함수' 자체를 학습**한다. 

- 특히, 이웃 노드 구조/특성을 이용해서 새로운 노드의 카테고리를 잘 예측하고, 완전히 새로운 그래프(단백질-단백질 네트워크)에서도 잘 동작함을 보인다. 

### 3.1 Embedding generation algorithm 
1. 초기화
	각 노드의 0단계 Embedding($h^0_v$)는 해당 노드의 Feature($x_v$)로 설정
2. 집계 및 임베딩 생성의 반복
	- for $k=1$ to $K$:
		a. 각 노드 $v$에 대해 이웃 $N(v)$의 $k-1$번째 $h^{k-1}_u$ 들을 집계(aggregate) →  $h^k_{N(v)}$   
	    b. 자신의 $k-1$ 번째 임베딩 $h^{k-1}_v$ 와 $h^k_{N(v)}$를 **concat**  
	    c. 이 벡터에 가중치 $W_k$, 비선형 $\sigma$을 적용해 $h^k_v$ 생성
- ( h^k_v )를 L2 정규화(벡터 길이 1로 만듦)
