Likelihood는 ‘어떤 일이 일어날 가능성’을 나타낸다. 
더 정확히 말하면 관측된 사건이 고정된 상태에서, 확률 분포가 변할 때(=확률 분포를 모를 때 = 가정할 때), 확률을 표현하는 단어이다. <br><br>
즉, 어떠한 관측값 $x$이 주어질 때, 변화되는 확률 분포 $p(\theta)$에서 주어진 관측값이 나올 확률이다. 
$$
\mathcal{L}(\theta|x) = p_\theta(x)
$$
다르게 말하면, “**모델 $p_\theta$가 관측된 $x$를 얼마나 잘 설명하느냐”를 수치화**한 것으로 볼 수 있다. <br>
<u>observed</u> $x$ 가 어떻게 생성됐는지 찾기 위해 결국 모델 내부에서 추정하는 여러 가능한 <u>latent</u> $z$를 모두 고려해야 진짜 $p_\theta(x)$를 얻을 수 있다. <br>![[스크린샷 2025-07-09 오후 1.54.22.png]]
그림과 같이 $x_0$를 관측할 확률 $p_\theta(x_0)$은, **이전 단계에서 $x_0$로 오는 모든 가능한 trajectory(latent $z$)를 marginalize**해야지 구할 수 있다. 그러나, 가능한 전 영역을 적분한다는 건 불가능하다. <br>
<b><font color="#e36c09">그렇지만,</font></b> $p_\theta(x_0)$<b><font color="#e36c09">의 Lower Bound를 Maximize할 수는 있다. </font></b>
$$
log\;p_\theta(x) \ge variational \; lower \;bound
$$
1. 원래 적분 식에서
$$
\log p_\theta(x) = \log \int p_\theta(x,z)\,dz
$$
2. $q(z|x)$를 분모 분자에 곱해준 뒤 정리하고, 
$$
\int p_\theta(x,z)\,dz
= \int \frac{q(z\mid x)}{q(z\mid x)}\,p_\theta(x,z)\,dz
= \int q(z\mid x)\,\frac{p_\theta(x,z)}{q(z\mid x)}\,dz
$$
3. 어떤 함수 $f(z)$에 대해 $\int q(z)f(z)dz = \mathbb{E}_q[f(z)]$ 이므로, 
$$
\int q(z\mid x)\,\frac{p_\theta(x,z)}{q(z\mid x)}\,dz
= \mathbb{E}_{q(z\mid x)}\;\!\!\Bigl[\tfrac{p_\theta(x,z)}{q(z\mid x)}\Bigr]
$$
4. jensen 부등식 적용하면, $log$를 기댓값 안으로 집어 넣을 수 있게 된다. 
$$
\log p_\theta(x)
= \log \mathbb{E}_{q(z\mid x)}\;\!\!\Bigl[\tfrac{p_\theta(x,z)}{q(z\mid x)}\Bigr]
\;\ge\;
\mathbb{E}_{q(z\mid x)}\;\!\!\Bigl[\log \tfrac{p_\theta(x,z)}{q(z\mid x)}\Bigr]
$$


### References
1. 