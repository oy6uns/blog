<b><font color="#de7802">prompt example</font></b>들 만으로 <b><font color="#de7802">parameter optimizing 없이</font></b> pretrained model을 여러 downstream task들에 잘 adapt 시키는 것을 <b><font color="#de7802">In-context Learning</font></b>이라고 한다. <br>LLM에 대해서는 이 능력이 입증 되었지만, graph에서는 어떻게 in-context learning을 수행할 지에 대한 연구는 아직 이루어지지 않았다. <br>본 논문에서는 **Pr**etraining **O**ver **D**iverse **I**n-Context **G**raph S**y**stems (**PRODIGY**) 를 소개한다. <br><font color="#de7802">First pretraining framework that <b>enables in-context learning over graphs</b></font>. 

- using novel prompt graph representation
	- which connects prompt examples and queries
- propose a GNN over the prompt graph and a corresponding family of in-context pretraining objectives

## Main Challenge: task representation의 통일성
- 다양한 그래프 기반 task(node-, edge-, graph-level)를 fine-tuning 없이 하나의 통일된 방식으로 표현할 수 있어야 한다. 
- **그래프 간 일반화**

## 제안된 기법
1. Prompt Graph : 그래프 기반 Task를 통일된 구조로 표현. 
2. PRODIGY: 다양한 그래프에 대해 in-context 학습이 가능한 프레임워크로, prompt graph 상에서 pretraining을 수행
![[스크린샷 2025-07-28 오후 2.52.04.png]]
### figure
> edge prediction task에 대해 설명한다. 
1. Original Graph $\mathcal{G}$에서 Prompt example $\mathcal{S}$를 우선 뽑는다. <br>$\mathcal{S}$ 는 head, tail 및 둘의 1-hop 이웃을 포함하는 subgraph이다. 
2. $\mathcal{S}$는 vector $v_x$로 encoding 되고, 해당 vector와 label을 가지고 모델이 학습하게 된다. 


# 3. Pretraining to Enable In-context Learning
## 3.1 Message Passing
위에서 만든 $\mathcal{S}$($\mathcal{G}^D$) 를 GNN Module(e.g. GCN, GAT)에 넣는다. <br>each node에 대한 embedding을 얻을 수 있다. 
$$
E ∈ \mathcal{R}^{|\mathcal{V}^D|×d} = M_D(\mathcal{G}^D)
$$
node classification task에 대해서는 <font color="#65b855">single node Embedding</font> $E_{\mathcal{V}_i}$를 $G_i$ for each data subgraph의 Embedding으로 설정한다. 
$$
G_i=E_{\mathcal{V}_i}
$$
link prediction task에 대해서는 <font color="#65b855">pair of node Embedding</font> 및 <font color="#65b855">all node representation의 Embedding의 Max-pooling 값</font>을 concatenate 한 값을 each data subgraph의 Embedding으로 설정한다. 
$$
G_i = W^T (E_{\mathcal{V}_1}∈\mathcal{V}_i ||E_{\mathcal{V}_2}∈\mathcal{V}_i ||max(E_i))+b
$$
where $||$ represents concatenation, $W ∈ \mathcal{R}^{3d×d}$ is a learnable weight matrix. 

이제 각 subgraph의 Embedding을 만들었으니, 각 subgraph의 Embedding들과 label node를 연결시켜 최종 Task Graph를 만들고, 이를 어텐션 기반 GNN 모듈 $M_T$에 입력으로 넣어준다. 
$$
H=M_T(G^T)
$$
결과적으로, $H \in \mathbb{R}^{|\mathcal{V}^T|\times d}$ 는 그래프의 모든 subgraph의 업데이트된 $d$차원의 벡터가 된다. <br>$$
O_i​=[cosine\;similarity(H_{x_i}​​,H_y​)]_{y∈Y}​
$$
<font color="#65b855">쿼리 노드에 대한 최종 subgraph의 벡터</font>와 <font color="#65b855">각 label 의 벡터</font>의 <font color="#65b855">cosine similarity를 비교</font>하여 가장 유사도가 높은 label을 선택해 쿼리 결과로 사용한다. <br>
- 예시: $\mathcal{y}=\{Red, Blue\}$라면, 
$$
O_i=\big[cos(H_{x_i}, H_{Red}), cos(H_{x_i}, H_{Blue})]
$$
  $O_i$에서 가장 큰 값(=가장 유사도가 높은 label)을 선택해 쿼리 $x_i$의 분류 결과로 사용한다. 

## 3.2 In-context Pretraining Objectives
<b><u>목표</u></b>:
- downstream task의 graph $G$와 독립적인 대규모 pretrain용 그래프 $G_{pretrain}$만으로, 
- <font color="#65b855">별도의 fine-tuning 없이</font> 그대로 few-shot <font color="#65b855">in-context learning이 가능한 모델이 키우기 위한 pretraining loss function</font>을 설계
### 3.2.1 Pretraining Task Generation
**두 가지 과제**를 수행한다. 
1. **어떤 노드가 같은 Local 이웃 그룹(subgraph)에 속하는지 분류**하는 self-supervised task
2. $G_{pretrain}$에 이미 있는 Node/Edge Label $f(x_i)=y_i$를 직접 활용하는 supervised task
<br>
---
#### 첫번째 과제, 
![[스크린샷 2025-07-29 오후 5.19.41.png]]
예로 위와 같은 그래프가 있다고 할 때, 
- 클래스 개수 $m=3$
- 학습할 sample 수 $k=2$
- 전체 쿼리 수 $n=3 → ⌈n/m⌉=1$
- k-hop $l=2$
로 정하고 계산해보자. <br>
#### 1) 기준 노드(클래스) 샘플링
$$
\{c1​,c2​,c3​\}=\{B,E,G\}\;\text{(무작위 샘플)}
$$
각 $c_i$가 하나의 “가짜 클래스”가 된다.
#### 2) $l=2$ 이웃 집합 구성
- $N_1 = \mathrm{Neighbor}(B,2)$
    - 1‑hop: ${A,C}$
    - 2‑hop: from $A→D$, from $C→D,E$ ⇒ $\{A,C,D,E\}$
    - 최종 $N_1=\{A,C,D,E\}$
- $N_2 = \mathrm{Neighbor}(E, 2)$
    - 1‑hop: $\{C,F,H\}$
    - 2‑hop: from $C→B,D$, from $F→G$, from $H→G$ ⇒ $\{C,F,H,B,D,G\}$
    - 최종 $N_2=\{B,C,D,F,G,H\}$
- $N_3 = \mathrm{Neighbor}(G, 2)$
    - 1‑hop: ${F,H}$
    - 2‑hop: from $F→E$; from $H→E$ ⇒ $\{F,H,E\}$
    - 최종 $N_3=\{E,F,H\}$
#### 3) 학습할 sample $S_i$ 샘플링
- $S_1\subset N_1=\{A,C,D,E\}$에서 2개:
$$
S_1=\{(A,B),\,(D,B)\},\quad y=B
$$
- $S_2\subset N_2=\{B,C,D,F,G,H\}$에서 2개:
$$
S_2=\{(C,E),\,(H,E)\},\quad y=E
$$
- $S_3\subset N_3=\{E,F,H\}$에서 2개:
$$
S_3=\{(E,G),\,(F,G)\},\quad y=G
$$
#### 4) 쿼리 $Q_i$ 샘플링($\lceil n/m\rceil=1$)
- $Q_1\subset N_1$에서 1개:  $Q_1=\{(C,B)\}$
- $Q_2\subset N_2$에서 1개:  $Q_2=\{(G,E)\}$
- $Q_3\subset N_3$에서 1개:  $Q_3=\{(H,G)\}$
합치면
$$
Q_{\rm NM} =\{(C,B),\;(G,E),\;(H,G)\}
$$

