<b><font color="#de7802">prompt example</font></b>들 만으로 <b><font color="#de7802">parameter optimizing 없이</font></b> pretrained model을 여러 downstream task들에 잘 adapt 시키는 것을 <b><font color="#de7802">In-context Learning</font></b>이라고 한다. <br>LLM에 대해서는 이 능력이 입증 되었지만, graph에서는 어떻게 in-context learning을 수행할 지에 대한 연구는 아직 이루어지지 않았다. <br>본 논문에서는 **Pr**etraining **O**ver **D**iverse **I**n-Context **G**raph S**y**stems (**PRODIGY**) 를 소개한다. <br><font color="#de7802">First pretraining framework that <b>enables in-context learning over graphs</b></font>. 

- using novel prompt graph representation
	- which connects prompt examples and queries
- propose a GNN over the prompt graph and a corresponding family of in-context pretraining objectives

## Main Challenge
task representation의 통일성
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
- node classification task에 대해서는 <font color="#65b855">single node Embedding</font> $E_{\mathcal{V}_i}$를 $G_i$ for each data subgraph의 Embedding으로 설정한다. 
$$
G_i=E_{\mathcal{V}_i}
$$
- link prediction task에 대해서는 <font color="#65b855">pair of node Embedding</font> 및 <font color="#65b855">all node representation의 Embedding의 Max-pooling 값</font>을 concatenate 한 값을 each data subgraph의 Embedding으로 설정한다. 
$$
G_i = W^T (E_{\mathcal{V}_1}∈\mathcal{V}_i ||E_{\mathcal{V}_2}∈\mathcal{V}_i ||max(E_i))+b
$$
  where $||$ represents concatenation, $W ∈ \mathcal{R}^{3d×d}$ is a learnable weight matrix. 

이제 각 subgraph의 Embedding을 만들었으니, 각 subgraph의 Embedding들과 label node를 연결시켜 최종 Task Graph를 만들고, 이를 어텐션 기반 GNN 모듈 $M_T$에 입력으로 넣어준다. 
$$

$$