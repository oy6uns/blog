---
date: 2025-12-27
created: 2025-12-27
modified: 2025-12-27
tags:
---
## typer 라이브러리
typer 라이브러리를 통해 사용하는 python command를 커스텀 할 수 있다. 
아래는 graphRAG에서 사용한 typer 방식이다. 
```python
import typer

app = typer.Typer(
    help="GraphRAG: A graph-based retrieval-augmented generation (RAG) system.",
    no_args_is_help=True,
)

@app.command()
def index(...):
    """Build a knowledge graph index."""
    # 인덱싱 로직
    
@app.command()
def query(...):
    """Query the knowledge graph."""
    # 쿼리 로직
```

![[스크린샷 2025-12-27 오후 5.25.44.png]]
오류 화면 자체도 이쁘게 나온다. 
커스텀할 수 있는건가??
![[스크린샷 2025-12-27 오후 5.27.57.png]]