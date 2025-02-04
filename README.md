# Ollama 엔진을 활용한 LLM 추론 모델 포팅 및 RAG 시스템 구축

![License](https://img.shields.io/badge/license-MIT-blue.svg)  
![Python](https://img.shields.io/badge/python-3.11+-green.svg)  
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20MacOS-blue.svg)

# 프로젝트 설명  
이 프로젝트는 **LLM 생성 모델을 활용하여 RAG 시스템을 도입해 특정 시나리오의 정보를 기억하고 추론에 활용할 수 있게 합니다.**

Ollama에서 DeepSeek-r1과 같은 llm 모델을 사전에 설치한 뒤에 소스코드에서 LLM 모델만 변경한다면, 사용할 수 있습니다.

## 📌 주요 기능
- **LLM 추론**: 사용자 요구에 맞게 Prompt를 정의하고 LLM 모델로 입력 후 답변 텍스트 생성
- **RAG 시스템**
    1) txt file loading
    2) text splitting
    3) vector_embedding
    4) retriever
    5) QA_PROMPT
    6) combine_documents_chain
    7) llm inference run!

## 📦 설치 방법
### 0. python 가상환경 생성 (venv or anaconda)

### 1. 필수 라이브러리 설치
    
    ```bash
    pip install -r requirements.txt
    ```

### 2. ollama 설치 (window 기준)

1. ollama window version 설치 [1]

2. ollama LLM model download (deepseek활용)
    
    ```python
    ollama run deepseek-r1:14b
    ```
    
3. Modelfile 작성

    예시
    ```
    # FROM LLM 모델 파일경로
    FROM sha256-6e9f90f02bb3b39b59e81916e8cfce9deb45aeaeb9a54a5be4414486b907dc1e
    ```
    
4. Modelfile 활용 ollama model create (local pc)
    - LLM model file과 Modelfile 파일을 같은 폴더 위치에 두고 그 경로로 CMD 접근
    - ollama create test -f Modelfile 명령어 실행
        - C:\Users\USER\.ollama\models 이 경로에 LLM 모델 파일이 설치됨

5. python 소스 코드 실행 [2]
    - 가상환경 생성, pip install ollama
    - python 코드 작성

6. Reference 
[1] https://ollama.com/
[2] https://github.com/ollama/ollama-python
