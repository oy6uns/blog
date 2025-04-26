---
tags:
  - "#python"
  - torch
---
여러개의 GPU 중 실험에 특정 GPU를 지정하여 실험을 돌리고자 할 때 아래 명령어를 사용하면 된다. 

GPU의 ID는 shell에서 
```shell
nvidia-smi
```
를 통해 확인 가능하다. 

즉, 아래 명령어는 **GPU #2**를 사용하여 **train.py** python file을 실행하라는 의미이다.
```shell
CUDA_VISIBLE_DEVICES={사용할 GPU ID} python train.py

# 2번 GPU 사용
CUDA_VISIBLE_DEVICES=2 python train.py
```