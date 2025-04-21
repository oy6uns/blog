# TGNN Taxanomy
## 1. Snapshot-based Methods (**스냅샷 기반**, STG 기반)
- 그래프가 일정 시간 간격(스냅샷)으로 분리되어 있고, 각 시점별로 정적인 그래프처럼 처리함.
- 시간적 연속성은 여러스냅샷 간의 관계를 모델링함으로써 학습됨.
- **하위 분류:**
    - **Model Evolution:**
        - GNN의 파라미터 자체가 시간에 따라 변화(진화)하도록 학습하는 방식. 예: EvolveGCN.
    - **Embedding Evolution:**
        - 각 노드 임베딩을 시간 축을 따라 순차적으로 업데이트(진화)하는 방식. 예: DySAT, VGRNN.

## 2. Event-based Methods (**이벤트 기반**, ETG 기반)
- 그래프 변화가 스냅샷이 아니라 **개별 이벤트(노드/에지의 추가, 삭제 등)** 수준에서 기록되고 모델링됨.
- 연속적이고 비정기적인 시점에서의 변화, 혹은 개별 이벤트가 시간적으로 중요한 경우에 강점.
- **하위 분류:**
    - **Temporal Embedding:**
        - 시간 임베딩(ex: Random Fourier Features 등)과 노드/에지 임베딩을 결합해, 순서 혹은 시간 차를 명시적으로 표현.
        - Attention 또는 RNN 계열로 시간 정보를 처리. 예: TGAT, NAT.
    - **Temporal Neighborhood:**
        - 각 노드의 이웃과 과거 이벤트를 모아두는 Mailbox/Memory 등 특수 구조를 사용해 동적으로 임베딩 업데이트.
        - 예: TGN, APAN, DGNN.

# 5. Temporal Graph에서의 Learning Task
다양한 학습 과제(task)와 학습 설정(setting)을 정리
어떤 종류의 예측이나 분류 문제를 풀 수 있는지, 그 문제들을 어떤 맥락에서 접근할 수 있는지를 체계적으로 설명

## 5.1 Learning Settings (학습 설정)
#### (1) Topological Setting 구조적 설정
1. Transductive Setting 
2. Inductive Setting
#### (2) Temporal Setting; 시간적 설정
1. Past Prediction
	- 누락된 정보 추정이나 과거 데이터 보완을 위해 활용되며, 
	  주변 시계열 패턴을 활용하여 예측한다. 
2. Future Prediction
	- 미래 패턴을 예측하는 것이 목적이다. 
→ 결과적으로 네 가지 조합(Transductive/Inductive × Past/Future)이 나올 수 있다. 

## 5.2 ==Supervised== Learning Tasks
1. Node Classificaiton
2. Edge Classification
3. (sub)Graph Classification
4. Regression
	위 분류 문제를 회귀로 변환
5. Link Prediction
	특정 시점에 **엣지가 생길 확률** 예측
6. Event Time Prediction
	**엣지(이벤트)가 언제 처음 발생**할지 그 시점 예측

## 5.3 Unsupervised Learning Tasks 
1. Clustering
	노드/그래프를 비슷한 특성을 바탕으로 군집화
2. Low-dimensional Embedding (LDE)
	노드/그래프를 **저차원 공간에 임베딩**하여 **시계열 변화 및 패턴을 시각화/분석**


# 6. 다양한 TGNN 방법론들
![[Assets/blog/content/GNN/Graph Neural Networks for temporal graphs; State of the art, open challenges, and opportunities/IMG-20250421200037.png]]
## 6.1 Snapshot-based Models
**“Snapshot”**: <span style="background:rgba(205, 244, 105, 0.55)">일정한 시간 간격</span>으로 전체 그래프(노드/엣지/속성 등)의 스냅샷을 연속적으로 나열한다. 
- 대표적 예시: 하루/1시간마다 찍힌 전체 네트워크 상태
### 6.1.1 Model Evolution
- **GNN 모델 파라미터 자체**를 시간에 따라 업데이트
- 대표 모델:
	- [EvolveGCN]: GCN의 파라미터를 RNN(LSTM, GRU 등)으로 시간에 따라 진화시킨다. ![[Assets/blog/content/GNN/Graph Neural Networks for temporal graphs; State of the art, open challenges, and opportunities/IMG-20250421200037-1.png]]
### 6.1.2 Embedding Evolution
- 각 시점의 노드 임베딩(h)을 RNN(혹은 attention)으로 시간축 상에서 직접 업데이트
$$h_{v(t_i)}=REC(h_v(t_{i-j}), \dots)$$
- 대표 모델:
	- [DySAT]: 구조적인 self-attention + 시간축 self-attention을 결합
	  → 즉, snapshot 내 + 시계열 간
	- [VGRNN]: 각 snapshot에서 VGAE → 이후, 임베딩을 LSTM등으로 진화
	- [SSGNN]: 시계열 + 공간정보 동시처리를 위한 reservoir 방식

## 6.2 Event-based Models 
**“Event”**: <span style="background:rgba(205, 244, 105, 0.55)">개별 노드/엣지의 생성/삭제/변화의 ‘이벤트’ 단위</span>로 시간 정보를 다룬다. 
- 즉, 스냅샷 없이 한 번에 한 이벤트씩 흐름을 추적
- 더 미세한/비정규 시간 단위 동적성을 표현
### 6.2.1 Temporal Embedding Methods
- 동적인(시간 가변) 그래프에서 노드 임베딩을 만들 때 “시간 정보”를 어떻게 효과적으로 반영할 것인가?
1. 시간 혹은 “시간 간격”(t-t’)을 임베딩으로 바꿈
2. 시간 임베딩 + 노드 임베딩을 결합
	- 노드 $v$의 임베딩 $h_v(t)$을 만들 때, “과거에 $v$와 연결된 이웃 $u$의 임베딩 $h_u(t’)$와, “그와 연결된 시간차 $g_{t-t’}$”를 함께 입력으로 사용
	- 이웃마다 연결된 시간이 다르므로 $g_{t-t’}$도 다르다!
$$ h_v​(t)=COMBINE((h_v​(t),g_​),AGGREGATE({(h_u​(t^′),g_{t−t^′​})}))$$
- $h_v​(t)$: 현 시점 t에서 v의 임베딩 (초기값 or 직전 값 등)
- $g_0​$: 현재시점 정보(또는 영벡터)
- ${(h_u​(t^′),g_{t−t^′}​)}$: **v의 이웃 u들이 t′에 v와 상호작용했던 그 시점의 임베딩과, 그때부터 지금까지 경과된 시간 임베딩 모음**
- $AGGREGATE$: 이웃별 정보를 attention 등으로 요약 (물리적 구조 + 시간 모두 반영)
- **$COMBINE$**: 자기 자신의 정보와 이웃으로부터 들은 정보를 결합

▶️ 즉, "시간+이웃" 정보를 동시에 활용! (이게 기존 GNN/layered-RNN과 다름)
- 대표 모델:
	- [TGAT]: Random Fourier Features Encoding을 사용해 각 edge마다 ‘시간차 임베딩’을 생성한다. 
	  (이웃 $u$가 $t’$에 나와 연결됐으면~) $[h_{u}(t'), g_{{t-t'}}]$을 attention의 입력으로 사용하여 여러 이웃 중 의미있는 이웃만을 강조

### 6.2.2 Temporal Neighborhood Methods (“Mailbox” 방식)
- ‘이벤트’가 생길 때마다(예: 두 노드가 연결/메시지를 주고받음 등) 해당 노드의 우편함에 **메시지(mail)** 가 저장된다. 
- 이 메시지에는 보통 1) 누구랑 연결됐는지 2) 어떤 type의 이벤트인지 3) 언제 일어났는지 4) 추가 정보(노드/엣지 특성 등) 등이 담겨 있다. 
1. 노드는 자신 mailbox에 있는 여러 메시지들을 Aggregate(예: mean, attention, LSTM 등)으로 요약
2. 그리고 지금 나의 임베딩 $h_v(t)$와 $Aggregate$한 것들을 Combine(예: sum, concatenate 등)한다. 
3. 그 결과가 최신 임베딩 $h_v(t)$!
$$h_v(t)=COMBINE(h_v(t),AGGREGATE(m_ϵ,ϵ=이벤트 로그))$$
▶️ “내가 최근에 누구와 어떤 종류의 상호작용/연결/이벤트를 가졌는가”를 자연스럽게 반영
- 대표 모델:
	- [TGN(Temporal Graph Networks)]: 각 노드마다 **메모리(memory)** 와 **메일박스(mailbox)** 를 모두 운영
	  이벤트 발생 시, mailbox로 메시지 날라오면
		- **memory** (과거 이력 반영하는 hidden-state)를 **업데이트**
		- 업데이트 후 **embedding**을 산출
		노드들의 mailbox는 병렬적으로 처리 가능 → 대규모 속도 및 효율성에서 뛰어남!
	- [DGNN]: 메일이 단순히 저장만이 아니라 이웃과의 **과거 상호작용 히스토리를 LSTM으로 처리**
	- [APAN]: 누가 나에게 메시지를 보낼지/받을지, 그리고 어떤 메시지에 더 집중해야 할지(중요도)를 attention으로 결정
	- [TGL]: 엄청나게 큰 그래프에서 최근 n개만 저장하여 memory/messaging 효율 강화
``` python
[Mailbox]   (메일함)
   |
[mail1]   <--- 이벤트1: A가 B랑 1분 전 연결됨
[mail2]   <--- 이벤트2: A가 C와 10분 전 연결
[mail3]   <--- 이벤트3: A가 D에서 데이터 받음
 ...
 ↓
aggregate(최근 메시지들)
 ↓
내 상태(h_A(t)) 최신화
```

# 7. Open challenges
## 1. 평가 (Evaluation) 문제

### ● 표준화의 부재
- **정적 GNN 연구**에서는 OGB(Open Graph Benchmark) 같은 표준 데이터·평가 패키지가 있음.
- 하지만 **==TGNN**에는 이런 표준화된 벤치마크와 프로토콜이 아직 없다.==
- 현재 각 연구는 각기 다른(작은/특화된) 데이터셋·성능 척도를 쓰기 때문에, TGNN 모델들의 실력을 ‘직접적으로’ 비교하기가 어렵다.
- 예: TGL 논문은은 수억 개의 엣지(초대형) 데이터셋을 만들었지만, 그 데이터에서 오직 자기 모델만 테스트함.

### ● 설명가능성(Explainability) 부족
- 기존 GNN에서는 ==설명가능성 연구==가 활발(왜 이 결과가 나왔는지), 그러나 TGNN에서는 극히 드문 상태임

## 2. 표현력(Expressiveness) 한계

### ● 기존 GNN의 표현력 연구
- 정적 GNN은 **WL(Weisfeiler-Lehman) 테스트**의 그래프 구분능력과 이론적 한계, 확장(Ring, k-WL 등)에 대한 연구가 많음.
- GNN이 무엇까지(몇 단계까지) 구분/학습할 수 있는가, 어떤 패턴은 못 배우는가가 잘 정립되어 있음.

### ● TGNN에는?
- **TGNN의 표현력/이론 연구는 매우 초창기**다!
    - 시간/동역학적 이웃의 의미 정의나 WL 테스트의 시간 확장 같은 근본적 난제가 많음.
- 일부 논문이 DTTG(이산시간 TG), ETG(이벤트 기반 TG)에 대한 동적 WL test 제안
    - 예: TGN이, 제안된 "Temporal WL 테스트"와 같은 수준의 표현력을 가진다고 증명함.
- 하지만:
    - **보편적/표준화된 "TGNN 표현력 이론"**,
    - 특히 **이벤트 기반 (연속적) TG**까지 완성된 것은 없음.
    - 고차원 구조(예: 2-WL 등)에 대한 이론도 부재

## 3. 학습 가능성(Learnability) 문제

### ● GNN의 학습 한계
- ==깊은 GNN에서 정보의 over-smoothing/over-squashing==(멀리서 온 정보가 모두 뭉개짐, bottleneck) 현상은 여전히 큰 문제.
- 해결책(드롭아웃, virtual node, neighbor sampling 등)도 아직 근본적이지 못함.

### ● TGNN에서의 난점
- TGNN에서는 이 문제에 ==시간축의 긴 의존성(long-term dependency)까지 더해져 한층 더 복잡해짐.==
    - 예: 오래 전에 있었던 중요한 이벤트의 영향도 잘 반영해야 함
- 실제로 안정적이면서도 깊고 복잡한(스냅샷/이벤트 많음) TGNN을 학습하는 기술은 아직 초기 단계
- 효과적이고, ==이론적으로 보장되는 ‘깊고 쉬운 학습법’이 필요==

## 4. 실제 활용(Real-world Applications)에서의 도전 과제

### ● 미해결 첨단 활용처들
- TGNN이 쓰일 법한 분야들은 많지만, 그간 너무 제한적이고, 아직 본격적으로 활용되지 못함:
    - **과학계산/물리 기반**:
        - Ex) 물리법칙과 신경망(Physics Informed NN) 결합, 시뮬레이션
        - 정적 GNN은 이미 성공 사례 많으나, TGNN이 본격 적용된 사례는 미비
    - **기후/환경/질병 등**:
        - TGNN의 시간+공간+관계 복합성 포착 능력이 특히 필요
        - 하지만 실제 대규모 예측/regression 과제엔 아직 소극적으로 적용됨
- ==**회귀 문제**(연속값 예측)에서 TGNN 응용은 거의 미개척== 상태