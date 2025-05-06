---
tags:
  - GNN
  - Temporal_GNN
  - Temporal_Graphs
  - Surver_Paper
---
> [paper link](https://www.arxiv.org/pdf/2406.09639)

# Abstact
- Multi-relational temporal graphs는 실제 데이터를 모델링할 때 매우 유용하며, 시간에 따라 변화하는 Entity들의 복잡한 상호작용을 잘 포착한다. 
- 여러 새로운 ML 모델이 제안되고 있지만, <span style="background:#d3f8b6">대규모이고, 표준화된 벤치마크 데이터셋, 그리고 신뢰성 있는 평가 방법의 부족</span>으로 인해 연구 발전에 제약이 있다. 
### Contribution
**Temporal Graph Benchmark 2.0(TGB 2.0)** 을 제안한다. 
- <span style="background:#d3f8b6">Temporal Knowledge Graph(TKG)</span> 및 <span style="background:#d3f8b6">Temporal Heterogeneous Graph(THG)</span> 상에서 <span style="background:#d3f8b6">link prediction을 평가</span>하는 데 특화된 벤치마킹 프레임워크이다. 
- TGB 2.0은 **8개의 새로운 대규모 데이터셋**(최대 5300만 개의 엣지 포함, 5개 도메인에 걸침)을 제공하며, 기존 데이터셋보다 훨씬 크다. 
- 단순히 임의의 음성(negative) 샘플이 아니라 edge(relation) type까지 고려해 음성 샘플을 생성하여 현실적 난이도를 갖추었다. 
### 실험을 통해 얻은 insight
1. **edge type 정보**를 적극적으로 활용하는 것이 그래프 예측 성능에 큰 도움이 된다. 
2. 단순한 heuristic 기반 Baseline이 종종 복잡한 최신 모델만큼 잘 동작한다. 
	→ **단순한 방법**이 복잡한 최신 모델에 비해 결코 만만하게 볼 수 없을 만큼 **성능이 꽤 높으니**, 반드시 같이 비교해야 한다. 
3. 대부분의 기존 방법들이 가장 큰 데이터셋에서는 아예 동작하지 못하거나 아예 느리다. 
	→ **확장 가능한(Scalability) 새로운 방법 연구**가 필요

# 2. Preliminaries
## Temporal ‘Knowledge’ Graph(TKG) 
- 정의: 시간의 흐름에 따라 변화하는 지식(관계)을 나타내는 그래프
- 구성요소: (subject, relation, object, timestamp)
- “subject가 언제 object와 어떤 관계를 맺는가”를 시간 정보까지 같이 저장
- **Temporal Triple** 또는 **Quadruple**이라는 표현을 쓴다. 

| 성분        | 의미                             | 예시                               |
| --------- | ------------------------------ | -------------------------------- |
| subject   | “주어” 역할을 하는 노드(Entity)         | Obama                            |
| relation  | 두 Entity 사이의 관계를 의미(Edge Type) | president_of, friend_with, likes |
| object    | “목적어” 역할을 하는 노드(Entity)        | USA, Michelle Obama, BasketBall  |
| timestamp | 이 관계가 발생한 시점(혹은 해당하는 시간 구간)    | 2009, 2012-04-01 00:00:00        |
## Temporal ‘Heterogeneous’ Graph(THG)
- 정의: TKG와 매우 유사하지만, 각 노드마다 고정된 Type(역할)이 지정되어 있음.
- 구성요소: (subject, relation, object, timestamp)에 더해 **node type function이 존재**

## Temporal Graph Forecasting
- 목표: 미래 시점(t+)에 나타날 새로운 링크(관계/행동/사건)를 예측하는 것
- 

