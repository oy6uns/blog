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

<br>
# 2. Preliminaries
## Temporal ‘Knowledge’ Graph(TKG) 
- 정의: 시간의 흐름에 따라 변화하는 지식(관계)을 나타내는 그래프
- 구성요소: (subject, relation, object, timestamp)
- “subject가 언제 object와 어떤 관계를 맺는가”를 시간 정보까지 같이 저장
- **Temporal Triple** 또는 **Quadruple**이라는 표현을 쓴다. 

```
Obama  --[president_of, 2009]-->  USA
```

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
- 예시 질의: (A, buys, ?, 2025) - A가 2025년에 무엇을 살 것인가?
- 방식: 정답 후보를 점수화해서(랭킹), 정답이 얼마나 위에 오는지 평가(Mean Reciprocal Rank 등) 
- TGB 2.0은 **Forecasting 문제를 중점적으로 다룸**

## Time Representations
1. <span style="background:#d3f8b6">Discrete(불연속/스냅샷)</span> 방식:
	- 시간별로 그래프의 전체 상태(스냅샷)가 주어짐. 
	- <span style="background:#d3f8b6">주로 TKG에서 사용</span>
2. <span style="background:#d3f8b6">Continuous(연속)</span> 방식:
	- 엣지(행동/관계)가 정확한 시점(초 단위 등)에 일어남. 
	- <span style="background:#d3f8b6">주로 THG에서 사용</span>
	- 더 많은 정보를 보존 가능하다. (필요시 불연속으로 변환도 됨)


> [!important] 종합
> TKG/THG는 모두 **(subject, relation, object, time)** 형태로 표현하지만, 
> - THG는 노드 타입이 추가되는 점이 다르고,
> - 최종 목표는 둘 다 **future relation(link) 예측** 이다. 

<br>
# 3. Related Work
## 1. Temporal Knowledge Graph (TKG) Methods
TKG 예측 방법들은 주로 두 가지 방식으로 발전하였다. 
1. 그래프 신경망(GNN) + 시계열 처리
	- 그래프 구조 정보(GCN, Message Passing)와 시간 흐름을 함께 다룸. (RNN)
	- 예시: RE-GCN, CEN 등.
2. 규칙 기반 또는 논리적 방법
	- 시간이 흐르면서 반복되는 논리적 패턴/규칙을 학습해서 미래를 예측. 
	- 예시: TLogic 등.

- 기타 접근법: 강화학습, Rule Mining 등.
- 대표 데이터셋: YAGO, WIKI, GDELT, ICEWS 등

## 2. Temporal Heterogeneous Graph (THG) Methods
THG 예측 방법들은 시간 표현 방식에 따라 분류된다. 
1. Continuous-time method(연속 시간 기반)
	- 이벤트 단위(초, 분 등)로 정보가 계속 추가됨. 
	- 예시: HTGN-BTW, STHN
2. Discrete-time method(불연속/스냅샷 기반)
	- 일정한 시간단위의 그래프 스냅샷(예: 월별, 연도별 등)으로 나눠서 처리함. 

- 대표 데이터셋: MathOverflow, Netflix, Movielens
	→ 여전히 소규모/한정적임. TGB 2.0에는 이보다 훨씬 큰 Myket, Github 등 포함

## 3. Graph Learning Benchmarks
최근에는 **OGB(Open Graph Benchmark)** 나 **TGB(Temporal Graph Benchmark)** 등,  
큰 규모의 다양한 벤치마크가 연구 커뮤니티에 큰 영향을 미치는 추세이다.
- OGB: <span style="background:#d3f8b6">정적</span>(시간 없는) 그래프 기준
- TGB: 시간 정보는 있으나, <span style="background:#d3f8b6">대부분 단일 관계(엣지 타입이 하나뿐)</span>
→ **TGB 2.0의 차별점:**  
    → “진짜 현실에서처럼 <span style="background:#d3f8b6">다중 관계</span>+시간+대규모 데이터”를 한 번에 다룬다!

<br>
# 4. Datasets
![[스크린샷 2025-05-07 오후 1.41.48.png]]
- **Inductive Test Nodes**: 테스트셋에만 나오는(학습셋에 없는) 노드 비율
- **Recurrency Degree**: ‘테스트셋에서 과거에 똑같은(혹은 비슷한) 관계가 등장했던 비율’을 측정 → 반복된 사건일수록 예측이 쉬움
- **Consecutiveness**: 연속된 시점에 같은 관계가 나타나는 정도(즉, 오래 지속되는 관계)
### 벤치마킹 목적의 데이터 분할
- 모든 데이터셋은 **시간 순서**대로 **학습:검증:테스트 = 70:15:15**로 분할  
	(즉, 미래 예측이 진짜 '미래'를 맞힐 수 있는 문제로 설정됨)

## 각 데이터셋 간략 소개
> 총 8개 데이터셋
### TKG (Temporal Knowledge Graph): 4개
- **tkgl-smallpedia**:  
    위키(Wikidata)의 엔티티들 중 100만 개 이하 ID의 엔티티 + 시간 정보를 가진 사실들만 추출. 미래의 사실 예측이 목표.
- **tkgl-polecat**:  
    전 세계 정치 행위자 간의 사건(협력, 적대 등) 기록 데이터. 2018~2022년의 시계열 정치 사건 예측.
- **tkgl-icews**:  
    국제 정치 사건 데이터(ICEWS)의 전체본. 1995~2022년 정치 이벤트 정보 포함. 이벤트(관계) 타입이 매우 다양.
- **tkgl-wikidata**:  
    Wikidata에서 더 많은 엔티티(3200만 개까지)와 과거~현재의 다양한 관계 및 시간정보까지 최대한 크게 구성.
### THG (Temporal Heterogeneous Graph): 4개
- **thgl-software**:  
    GH Arxiv 기반의 Github 활동 데이터. 사용자/PR(풀리퀘)/이슈/레포 등의 다양한 노드 및 여러 관계와 시간정보.
- **thgl-forum**:  
    Reddit 포럼의 사용자/서브레딧 간 상호작용 데이터.
- **thgl-myket**:  
    안드로이드 앱 마켓에서 유저-앱 설치/업데이트 시계열 상호작용(6개월, 130만 유저, 2억여 인터랙션).
- **thgl-github**:  
    GH Arxiv에서 추출, 2024년 3월 기준의 또 다른 대규모 Github 활동 데이터.

![[스크린샷 2025-05-07 오후 2.14.29.png]]
### TKG vs. THG 
- TKG(지식그래프): coarse(연 단위) 단위라 한 시점에 이벤트 수 많음, <span style="background:#d3f8b6">과거에는 적고 최근으로 갈수록 더 많아짐(=디지털화 진행)</span>
- THG(이종 그래프): 초 단위로 매우 세분화, 한 시점당 이벤트 수가 적고, <span style="background:#d3f8b6">특정 시간대에만 급증(피크) 등 고유한 패턴</span>

# 5. Experiments 
## 1) Evaluation Protocol
미래에 어떤 관계(Link)가 나타날지 예측, <span style="background:#d3f8b6">'정답이 위쪽에 오도록'</span> 랭킹을 매김
### 평가 지표(Metric)
MRR(Mean Reciprocal Rank)
- 정답이 리스트에서 몇 번째에 나오는지의 역수를 평균 내는 지표 (1등이면 1, 2등이면 0.5, 3등이면 0.33 ...)
- 동일 시점에 정답 후보가 여러개라면, 가장 상위 정답으로 계산
### Negative Sampling
> [!notice] Example
> 예를 들어 페이스북 같은 소셜 그래프가 있다고 가정해보자. 
> - 노드 = 사람  
> - 엣지 = "친구다(friend_of)" 관계
- **작은 데이터셋**에서는 **1-vs-all 방식 사용**이 가능하다. 
	- 전체 사람 수가 적으니까 **모든 후보를 전부 검사할 수 있다.** 
- 그러나, **큰 데이터 셋**에서는 <u>모든 사람과 비교를 하는 것이 불가능</u>하다. 
	- 친구 후보를 “비슷한 타입”으로 제한한다. 
	- 예를 들어, 나와 **같은 나라/학교/관심사** 가진 사람 중에서만 가짜 후보를 뽑는다. 
> [!important] 결론
> - **작은 그래프** = 후보 전원 검사 (완벽하지만 데이터 작을 때만 가능).
> - **큰 그래프** = 비슷한 애들만 골라서 비교 (현실적이고 어려운 상황 반영).

## 2) 실험 세팅 및 비교 
#### **(1) TKG 실험**
- 비교 모델:
    - RE-GCN, CEN, TLogic (대표 딥러닝/논리 기반 최신 TKG 예측모델)
    - 휴리스틱 베이스라인: Recurrency Baseline, EdgeBank(과거에 나왔던 관계를 점수화, 단순반복/윈도우 방식)
- 실험 결과:
    - **특정 환경(특히 반복 많은 데이터)** 에서는 복잡한 모델도 단순 휴리스틱보다 크게 나아지지 않거나 거의 비슷함
    - EdgeBank는 멀티-관계 정보 안 쓸 때 성능이 낮아지고
    - 대형 데이터셋(tkgl-wikidata 등)은 **EdgeBank**만 동작, 나머지는 메모리·시간 초과 → 확장성 문제 심각!
    - Recurrency Degree(관계의 반복이 많을수록) 성능이 좋고, <span style="background:#d3f8b6">반복이 적은 관계에서는 개선 여지가 많음</span>

#### **(2) THG 실험**
- 비교 모델:
    - TGN (오리지널/엣지타입 추가 버전): 단일 관계 기반 딥러닝 모델을 멀티-관계에 맞게 개조
    - **STHN**: 연속 시간·관계 정보를 잘 활용하는 최근 방법
    - Recurrency Baseline, EdgeBank
- 실험 결과:
    - STHN 등 최신 딥러닝, 엣지 타입을 활용한 모델이 확실하게 우수
    - 하지만 **STHN(복잡한 그래프 딥러닝)은 대형 데이터에서는 메모리만 185GB 넘게 필요, 스케일 불가**
    - TGN 기본 버전도 확장성 한계, 엣지 타입 추가 시 성능 개선 뚜렷
    - **EdgeBank(단순 규칙 기반)** 는 계속 확장 가능하지만 예측력은 제한적

> [!summary] 읽어볼 논문
> - **EdgeBank** (Both)
> - **RE-GCN, TLogic** (TKG)
> - **TGN, STHN** (THG)

> [!check] 실험 결과 요약/분석
> - **엣지(관계) 타입 정보를 잘 활용**하면 성능이 확 좋아짐
> - **단순 휴리스틱(반복 기반 등)도 의외로 성능이 좋다**
> - **대형 데이터셋에서는 최신/복잡한 모델들은 실제로 돌리기 힘들 만큼 느리거나 메모리 초과되는 경우가 많다**

## 3) 실험 결과
### Temporal Knowledge Graph Link Prediction Results
![[스크린샷 2025-05-07 오후 3.30.01.png]]
### Temporal Knowledge Graph Link Prediction Results
![[스크린샷 2025-05-07 오후 3.30.51.png]]
<br>
# 6. Conclusion
1. **TGB 2.0**은 **다중 관계와 시간 정보를 가진 그래프(Temporal Knowledge Graph, TKG & Temporal Heterogeneous Graph, THG)** 학습용으로 
	- **대규모, 다양한 도메인의 8개 새 데이터셋**을 제시했고, 
	- **공정·현실적·재현 가능한 평가환경**(다운로드/처리/자동평가 및 리더보드)까지 제공한다. 
2. **기존 알고리즘들은** 대규모 데이터셋에서는 **매우 느리거나, 메모리 제한 때문에 아예 실행조차 못하는 한계**가 있다. 
    - 미래 연구 방향은 **정말 '확장 가능한(scalable)' 방법 개발이 필수**라는 것!
3. 한계
	- 이번 벤치마크는 **THG 데이터에 한해 연속 시간(continuous-time) 세팅만 지원**한다. 
    - discrete(불연속) 시간 세팅을 지원하면 다양한 방법 비교가 더 잘 이루어질 수 있을 것.
	- 아직 데이터 도메인은 5개로 제한(생명과학, 논문 인용 등 미포함)

# Appendix
## Dataset Details
### 파일 구조 및 구성
#### (1) **TKG (Temporal Knowledge Graph) 데이터셋**
- **edgelist.csv** :  
    시간 정보가 포함된 링크(관계)를 (subject, relation, object, timestamp) 형태로 저장
- **val_ns.pkl / test_ns.pkl** :  
    검증·테스트용 네거티브 샘플(음성 샘플) 사전 생성 파일
- **static_edgelist.csv** :  
    (tkgl-smallpedia, tkgl-wikidata 전용) 시간정보 없는 정적 관계 정보도 함께 제공

#### (2) **THG (Temporal Heterogeneous Graph) 데이터셋**
- **edgelist.csv** :  
    시간정보, 관계타입·노드타입 포함된 엣지 정보
- **val_ns.pkl / test_ns.pkl** :  
    검증 및 테스트 음성 샘플 파일
- **nodemapping.csv / edgemapping.csv** :  
    노드·엣지 이름, 타입 정리
- **nodetype.csv** :  
    각 노드의 타입(역할 등) 설명

## 실험 Hyperparameter 설정
![[스크린샷 2025-05-07 오후 4.23.05.png]]
