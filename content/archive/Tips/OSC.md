---
date: 2026-01-21
created: 2026-01-21
modified: 2026-01-21
tags:
---
- /fs/ess/PAS1289
	- 공용 file system 있는 폴더

## Demo - 간단한 작업 1시간 동안만 실행
```bash
sinteractive -N 1 -n 1 -c 4 -t 01:00:00 --g 1 -A PAS1289 -J llm_dev
```
- `-N` : 1개 노드
- `-n`: 1개 작업
- `-c`: 4개 cpu
- `-t`: 1시간
- `-g`: 1개 gpu
- `-A`: account 정보(이름)
- `-J`: Job 정보 

## GPU 오래 잡고 돌리기 
```bash
mkdir -p logs
sbatch run.sh
```
### 현재 queue 상태 확인
```bash
squeue -A PAS1289
```
### 실행 로그 확인
```bash
tail -f logs/{현재 file log 파일명}.out
```

## 실행 예시
- `module purge`: 지금 세션에 로드돼 있는 모듈들을 전부 내리고(초기화) 깨끗한 상태로 시작합니다.
- `module load miniconda3/24.1.2-py310`: 클러스터가 제공하는 Miniconda3(파이썬 3.10 포함) 모듈을 로드해서 conda 명령을 쓸 수 있게 합니다.
- `conda info --envs: conda`가 인식하는 가상환경 목록을 출력합니다(디버깅용).
- `eval "$(conda shell.bash hook)`": 배치 스크립트 같은 비대화형 bash에서도 conda activate가 동작하도록, conda의 쉘 초기화 코드를 현재 쉘에 주입합니다.
- `conda activate myenv_pip`: myenv_pip라는 conda 환경을 활성화해서, 이후 명령들이 그 환경의 파이썬/패키지(여기선 pip로 설치한 PyTorch)를 사용하게 합니다.

```bash
#!/bin/bash
#SBATCH -A PAS1289
#SBATCH -p nextgen
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -G 1
#SBATCH -t 06:00:00
#SBATCH -J example
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
  
set -euo pipefail
module purge
# Ascend: anaconda3 모듈 대신 miniconda3가 제공됩니다
module load miniconda3/24.1.2-py310

conda info --envs
eval "$(conda shell.bash hook)"
conda activate myenv_pip  

nvidia-smi
python mlp.py
```

아마 6시간이 `#SBATCH -t` 최대인거 같은데, 
6시간보다 적게 설정해놨다면, `scontrol update JobId=<JOBID> TimeLimit=06:00:00`
명령어를 통해 시간을 연장할 수 있다. 

먼저 끝나면 그 즉시 job은 종료되고 큐에서 사라진다. 