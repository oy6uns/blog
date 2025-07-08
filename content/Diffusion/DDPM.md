>https://www.youtube.com/watch?v=uFoGaIVHfoE
>위의 영상을 참고하여 작성한 게시글입니다


> [!idea] Idea
>작은 sequence에서의 확산은 forward와 reverse **모두 가우시안**일 수 있다.
  작은 공기 입자의 다음 위치는 가우시안 분포 안에서 결정될 수 있다.

## forward process

$$ \mathcal{N}(x_t;\;\sqrt{1-\beta_t}x_{t-1}, \beta_tI) $$

$$ x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon $$

여기서
- $x_{t-1}$과 $\epsilon$은 독립이다.
- $Var(x_{t-1})=1, Var(\epsilon)=1$ 라 가정한다.
두 확률변수 $A$와 $B$가 독립이고, 상수 $a, b$ 가 있을 때 
(공분산 항 $2ab\;Cov(A, B)$는 독립이면 0이 되기에)

$$ Var(aA+bB)=a^2Var(A)+b^2Var(B) $$

$$ Var(x_t)=(\sqrt{1-\beta_t})^2\cdot1+(\sqrt{\beta_t})^2\cdot1=(1-\beta_t)+\beta_t=1. $$

결과적으로, 두 분산이 정확히 더해서 1이 되게 된다. 이 덕분에 **매 스텝마다 전체 분산이 일정하게 유지**된다!
![[스크린샷 2025-07-08 오후 12.56.48.png]]
DDPM에서는 **특정한 시점 $t$에서의 노이즈를 바로 샘플링할 수 있게끔** 수식적 기반을 마련해두었다.
![[스크린샷 2025-07-08 오후 12.57.18.png]]
계속해서 노이즈를 더해나가서, 최종적으로 $q(x_T)$ 라는 가우시안 노이즈가 나오게끔 한다.