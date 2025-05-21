# 1. Introduction
## 기존 추천시스템
- 예전 추천시스템(Collaborate Filtering)은 주로 **정적인 사용자-아이템 관계**만 본다. 
	- 즉, “A가 B를 샀으니 비슷한 C도 살 거다” 처럼 한 시점의 “취향”만 반영
- 현실에서는 사용자의 **취향이 시계열적으로(순차적으로) 변함**. 예를 들어, 영화/음악/상품 취향이 최근에 본 것(들은 것, 산 것)에 영향을 받음.
- 그래서 최근에는 사용자의 <span style="background:rgba(208, 235, 166, 0.55)">Sequential Information</span>을 반영하는 연구가 많아졌다. 
	- Markov chain, RNN/LSTM/GRU, CNN 및 Attention 기반 연구

## 기존 접근법의 공통적인 한계점
- **단일 사용자의 시퀀스만**(A의 구매, 클릭, 시청 기록) 집중적으로 보고, **”다른 사용자들과의 동적 협업 신호(동시적, 혹은 유사한 시점의 연관된 행동들)"** 는 잘 고려하지 못함.

## 제안한 방법
- **모든 사용자 시퀀스**를 **동적 그래프**에 통합(엣지에 타임스탬프/순서 정보 부여).
- **Dynamic Graph Recommendation Network(DGRN)** 라는 새로운 GNN 구조를 설계해, 이 서브그래프에서 사용자의 장-단기 선호를 추출.
- **다음 아이템 예측**은 '동적 그래프 내에서 사용자 노드와 아이템 노드가 새로 연결될 확률'을 구하는 **링크 예측 문제**로 변환.

# 2. Related Work
## 기존 방법론
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



