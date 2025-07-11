---
date: 2025-03-13
created: 2025-03-13
modified: 2025-03-13
tags:
---

Docker를 사용하면 **어떤 시스템에서도 동일한 환경을 그대로 재현이 가능**합니다.

- 한번 만든 환경을 다른 사람과 쉽게 공유하고 동일하게 실행이 가능하기에 → **손쉬운 배포 및 공유가 가능하고**
- 각 **라이브러리들의 버전 관리에도 용이**합니다.

✅ **결과적으로, 개발 및 실험 환경을 누구나 동일하게 실행할 수 있도록 보장하여 재현성을 극대화할 수 있습니다!**

# 1. SSH 서버 접속 하기

ssh 서버 주소 [서버 url] 에 계정 생성이 완료되었다면,

vscode 새창을 킨 다음 위의 검색 창에서 >Remote-SSH: Connect to Host… 로 들어가줍니다.
![[Pasted image 20250427175056.png]]
아래의 configure SSH Hosts… 를 클릭합니다.
![[Pasted image 20250427175112.png]]
/Users/{본인의 맥북 id}/.ssh/config 로 된 파일을 클릭합니다.
![[Pasted image 20250427175123.png]]
아래와 같이 입력해줍니다.

⭐️ User에는 [서버 url] 계정 생성 시의 ID 값을 넣어야 합니다.
![[스크린샷 2025-04-27 오후 5.53.03.png]]
이후 저장을 눌렀다면,

검색 창에서 >Remote-SSH: Connect to Host… 로 다시 들어갔을 때,

`vibaLab` 이 추가된 것을 확인할 수 있습니다.
![[Pasted image 20250427175440.png]]
클릭하여 SSH 서버로 접속하면 아래와 같은 화면을 볼 수 있습니다.

1. Open Folder
2. OK

를 차례대로 눌러줍니다.
![[Pasted image 20250427175453.png]]

<span style="background:#d3f8b6">- - -</span>
### ➕ 로컬 RSA 키 SSH 서버 등록 방법

> 암호 없이 SSH 서버 접속하기!

1. 로컬에 SSH 키가 있는지 확인  
```java
	ls -l ~/.ssh/id_rsa.pub
```
2. 없다면, RSA 키를 새로 생성
```java
    ssh-keygen -t rsa
```
- `-t rsa` : RSA 타입의 키 생성
	🔹 이제 ~/.ssh/id_rsa.pub 파일이 생성됨!
3. SSH 서버에 공개 키 추가
```c
    ssh-copy-id user@remote-server
```
- user → SSH 서버의 사용자 이름
- remote-server → SSH 서버의 IP 주소 또는 도메인 

# 2. docker image build

### Docker Extension 설치

![[Pasted image 20250427175737.png | 300]]

vscode 창 좌측의 Extension tab에 들어가서

container를 검색하여 나오는

1. Docker
2. Dev Containers

extension을 둘 다 설치해주어야 합니다!

### Dockerfile 생성 및 정의

다시 파일 탐색기로 돌아와서

2개를 새로 만들어주어야 합니다.

1. **Dockerfile**: Docker image build 를 어떻게 할지를 정의합니다.
2. **workspace 폴더**: Docker Container의 파일들을 Mount시켜줄 폴더를 새로 만들어줍니다.

![[Pasted image 20250427175824.png]]

완료되었다면 **Dockerfile** 을 열어 아래 코드를 작성해줍니다.

```bash
# 1. 어떤 docker image를 사용하여 build를 할지 -> 여기서는 다운 받아둔 NVIDIA PyTorch 공식 이미지를 활용한다. 
\\FROM nvcr.io/nvidia/pytorch:25.02-py3

# 2. 기본 패키지 설치
RUN apt-get update && apt-get install -y vim git wget graphviz

# 3. 기본적으로 나는 GNN을 사용할 것이기에 GNN관련 라이브러리들에 대해 정의해주었다. 
# 추가적으로 필요한 라이브러리가 있다면 명시해주면된다. 

# PyTorch Geometric(PYGRAPHS) 관련 필수 라이브러리 설치
RUN pip install --no-cache-dir torch torchvision torchaudio \\
    numpy pandas matplotlib scikit-learn tqdm networkx \\
    torchmetrics rdkit

# PyTorch Geometric (PyG) 설치
RUN pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv \\
    torch-geometric -f <https://data.pyg.org/whl/torch-2.1.0+cu121.html>

# Deep Graph Library (DGL) 설치
RUN pip install --no-cache-dir dgl -f <https://data.dgl.ai/wheels/cu121.html>

# 4. optional 하긴 하지만, chrome 창에서 jupyter notebook을 실행하고 싶을 때를 위해 추가해두었다. 
\\CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
```

### Dockerfile Build

**Dockerfile**을 저장한 뒤에, 현재 `Dockerfile`이 있는 디렉토리에서 터미널을 열고 아래의 shell 명령어를 실행해줍니다.

```bash
docker build -t my-env .
```

✅ `-t my-env` → 빌드된 이미지를 `my-env`라는 이름으로 저장

✅ `.` → 현재 디렉토리에 있는 `Dockerfile`을 사용하여 빌드

### Build 확인

`my-env`라는 이름의 이미지가 잘 생성되었는지 확인해줍니다.

```bash
docker images
```

# 3. docker container attach

### Dockerfile Run

> 빌드된 Docker 이미지 위에 **Container 환경을 생성**하여 실행합니다.

컨테이너 내부에서 두 개의 폴더를 생성하고, 각각 필요한 위치에 **마운트(Mount)** 할 예정입니다.

1️⃣ **현재 Docker 환경(Local) 폴더**
- 기본적으로 모든 작업이 진행되는 주요 작업 디렉토리입니다.

2️⃣ **NAS(Network Attached Storage) 폴더**
- 대용량 데이터를 저장하거나 활용할 때 유용하게 사용할 수 있습니다.

### NAS에 Mount를 위한 폴더 생성
> 터미널에서 직접 NAS 폴더로 접속하여 생성해도 무관합니다!

![[스크린샷 2025-04-27 오후 6.00.09.png]]
1. [NAS IP] 로 NAS 접속
2. /viba_shared/home 안에 본인이 원하는 이름의 폴더 생성
    → 여기서는 ‘honggildong’ 이라는 이름으로 만들었다 가정하고, 설명을 진행하겠습니다.

다시 터미널로 돌아와서, 아래와 같은 shell 명령어를 실행해줍니다.

```bash
docker run --gpus all -d -it --name my-container \\
    -v $(pwd)/workspace:/workspace/Local \\
    -v /nas/home/honggildong:/workspace/NAS \\
    my-env
```

| 옵션                                        | 설명                                                                                            |
| ----------------------------------------- | --------------------------------------------------------------------------------------------- |
| `docker run`                              | 새로운 컨테이너를 실행하는 명령어                                                                            |
| `--gpus all`                              | 컨테이너에서 **모든 GPU를 사용 가능**하도록 설정                                                                |
| `-d`                                      | 컨테이너를 **백그라운드(detached mode)에서 실행** (터미널 종료해도 계속 실행됨)                                         |
| `-it`                                     | **인터랙티브(Interactive) 모드 + 터미널(TTY) 모드**로 실행 (즉, 명령어 입력 가능)                                    |
| `--name my-container`                     | 컨테이너 이름을 `my-container`로 설정                                                                   |
| `-v $(pwd)/workspace:/workspace/Local`    | **현재 로컬 디렉토리** 내 `workspace` 폴더를 컨테이너 내 `/workspace/Local`로 마운트 (데이터 공유)                      |
| `-v /nas/home/honggildong:/workspace/NAS` | **NAS(Network Attached Storage) 경로** `/nas/home/shbae`를 컨테이너 내 `/workspace/NAS`로 마운트 (데이터 공유) |
| `my-env`                                  | 실행할 **Docker 이미지 이름** (이전에 빌드한 `my-env` 이미지 기반으로 컨테이너 생성)                                     |

![[스크린샷 2025-04-27 오후 6.01.08.png]]
⬆️ 이런식으로 떴다면 성공입니다!

<span style="background:#d3f8b6">- - -</span>
### ➕ 생성된 container 삭제

만약, 추가적인 옵션을 넣고 싶거나, 오타를 내서 다시 `docker run`을 해야하는 상황이라면,

아래와 같은 명령어를 차례로 입력해줍니다.

```bash
#1. 실행 중인 container 중지
docker stop my-container

#2. 해당 container 삭제
docker rm my-container
```

### 생성된 컨테이너 확인 및 접속

실행 중인 컨테이너 목록을 확인해줍니다.

```bash
docker ps
```

![[Pasted image 20250427180223.png]]
위와 같이 우리가 정의한 docker image 이름으로, 정의한 이름의 docker container 가 있다면 성공입니다!

이제 마지막으로 vscode 검색창에

**>Dev Containers: Attach to Running Container…**

를 클릭하면 생성한 Container로 접속할 수 있습니다!

![[Pasted image 20250427180240.png]]

### 완료!
![[Pasted image 20250427180251.png]]
접속한 Docker Container에서

1. 기본적으로 **Local 폴더**에서 작업을 하면 되고
2. **NAS 폴더**는 필요한 상황(**용량이 큰 파일 업로드, 다른 사람들과 공유하고픈 파일**) 에 사용하면 됩니다~~

### +@ 동작하고 있는 컨테이너 리소스 확인

```bash
docker stats
```

![[Pasted image 20250427180301.png]]