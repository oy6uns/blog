---
date: 2025-07-21
created: 2025-07-21
modified: 2025-07-21
tags:
---
> Exploitation of a Latent Mechanism in Graph Contrastive Learning: Representation Scattering (NeurIPS’ 24)
> https://openreview.net/pdf?id=R8SolCx62K

Graph Contrastive Learning (GCL) 은 manual annotation 없이 graph representation을 생성하는데 효과적으로 알려져왔다. <br>일반적인 GCL은 3가지 main frameworks로 나뉜다: 
1. Node Discrimination 
2. Graph Discrimination
3. Bootstrapping
3가지 main frame work 모두 좋은 성능을 달성했지만, 근본적인 메커니즘과 그 요인은 아직 완전히 파악되지 않았다. <br><br>본 논문에서는 3가지 main frameworks를 revisiting하고, seemingly disparate(겉보기에 달라보이는) 3가지 방법들을 통합하여 더 높은 성능을 가지는 general mechanism인 <b><font color="#de7802">‘representation scattering’</font></b>에 대해 소개한다.<br><br>GNN이나 일반 딥러닝 모델에서 학습된 표현(embedding)이 너무 비슷

