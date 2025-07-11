- parameterized Markov chain trained using variational inference
- Transition of this chain are learned to reverse a diffusion preocess
- **Markov chain gradually adds noise** to the data opposite direction of sampling

### Sampling? 
완전한 노이즈 한 점에서 출발해, 역방향 $p_\theta(x_{t-1}|x_t)$를 반복하여 노이즈를 점점 제거해 최종적으로 새로운 $x_0$를 생성하는 과정이다. 

### Contribution
- has been no demonstration yet but this paper shows diffusion models **actually are capable of generating high quality samples** (Section 4) 
- **denoising score matching over multiple noise levels during training** <br>equal to **annealed Lagevin dynamics during sampling** (Section 3.2) 
- obtained our best sample quality results (Section 4.2) 

### Denoising step
$$
N(x_{t−1}​;μ_θ​(x_t​,t),Σ_θ​(x_t​,t))
$$
- $\mu_\theta(x_t, t)$ : 모델이 예측한 평균(mean)
- $\Sigma_\theta(x_t, t)$: 모델이 예측한 공분산(covariance)
노이즈 낀 $x_t$에서 이전 단계의 $x_{t-1}$는 평균 $\mu_\theta$, 분산(or 공분산) $\sigma_\theta$인 가우시안 분포에서 뽑게끔 학습

- What <b><font color="#e36c09">distinguishes diffusion models from other types of latent variable models</font></b> is that the approximate posterior $q(\mathbf{x}_{1:T}|x_0)$, called the <b><font color="#e36c09">diffusion process or forward process, is fixed to a Markov chain</font></b><br>Diffusion model이 다른 모델들과 다른 점은 forward process를 고정된 Gaussian Markov chain으로 정해놓고, backward만 학습한다는 점이다. 

### TODO
- 3.2절 reverse process 수식 전개 부분 다시 봐보기
- 3.3절 Data Scaling 부분 다시 봐보기
- DDIM, paper reading