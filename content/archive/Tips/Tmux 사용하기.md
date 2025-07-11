---
tags:
  - shell
date: 2025-04-02
created: 2025-04-02
modified: 2025-04-02
---
tmux 세션 위에서 실험을 하게 되면, 세션을 끄지 않는 이상 백그라운드에서 계속 돌게 된다. 노트북을 닫아도 실험이 돌아가게끔 할 수 있다!

```shell
# shell script 실행
tmux new -s {세션 이름} -d './{쉘 스크립트 파일명}.sh' 

# python 파일 실행
tmux new -s {세션 이름} -d 'python {실행할 python 파일명}' 
```

## 실행 중인 세션 확인
```shell
tmux ls
```

## 특정 세션으로 들어가기
```shell
tmux attach -t {세션 이름}
```

## 세션에서 빠져나오기(계속 실행되게 두고)
```shell
Ctrl + b 누르고, d
```

## 세션을 종료하려면?
```shell
tmux kill-session -t {세션1 이름}
tmux kill-session -t {세션2 이름}
...
tmux kill-session -t {세션n 이름}
```
또는, 
```shell
tmux kill-server # 모든 세션 강제 종료
```

