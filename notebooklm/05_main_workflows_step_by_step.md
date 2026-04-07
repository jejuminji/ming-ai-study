# 주요 사용 흐름을 단계별로 보기

## 이 문서의 목적
이 레포를 실제로 어떻게 읽고 활용하면 좋은지, 초보자 기준으로 단계별 사용 흐름을 정리합니다.

## 추천 독자 수준
beginner

## 아주 짧은 요약
처음 공부할 때, 용어가 막힐 때, 예제를 볼 때, NotebookLM에 올릴 때의 흐름을 따로 보면 훨씬 덜 헷갈립니다.

## 워크플로우 1. 처음 공부를 시작하는 흐름

### 목적
레포 전체의 큰 그림을 먼저 잡는다.

### 시작 조건
- AI를 처음 공부한다.
- 어디부터 봐야 할지 모르겠다.

### 입력
- `README.md`
- `docs/00_AI_roadmap.md`
- `docs/01_AI_vs_ML_vs_DL.md`

### 처리 과정
1. `README.md`로 이 레포의 목적과 읽기 순서를 본다.
2. `docs/00_AI_roadmap.md`로 전체 지도부터 본다.
3. `docs/01_AI_vs_ML_vs_DL.md`로 가장 기본적인 구분을 잡는다.
4. 이해가 안 되는 용어는 바로 `docs/10_glossary.md`에서 확인한다.

### 결과
- AI, ML, DL의 관계가 잡힌다.
- "무엇을 먼저 공부해야 하는지" 감이 생긴다.

### 어디서 문제가 날 수 있는가
- LLM부터 바로 보려 하면 기초가 약해질 수 있다.
- 용어를 한 번에 너무 많이 보려 하면 부담이 커질 수 있다.

## 워크플로우 2. 신경망 기초를 잡는 흐름

### 목적
딥러닝과 LLM을 보기 전에 신경망 기초를 잡는다.

### 시작 조건
- 딥러닝이 막연하다.
- weight, bias, backpropagation 같은 단어가 낯설다.

### 입력
- `docs/04_deep_learning_basics.md`
- `docs/13_ai_learning_algorithms.md`
- `docs/14_neural_network_training_and_classification.md`
- `docs/15_classic_deep_learning_architectures.md`

### 처리 과정
1. `04`에서 신경망과 딥러닝의 큰 그림을 본다.
2. `13`에서 GD, SGD, Adam, PPO, CQL 같은 알고리즘 이름을 정리한다.
3. `14`에서 weight, bias, sigmoid, softmax, MLE, backpropagation을 본다.
4. `15`에서 CNN, RNN, GAN, 깊은 신경망의 한계를 본다.

### 결과
- LLM 이전의 딥러닝 기초가 잡힌다.
- 용어가 단독 단어가 아니라 흐름으로 연결된다.

### 어디서 문제가 날 수 있는가
- 수학 공식을 다 이해해야 한다고 느끼면 부담이 커질 수 있다.
- CNN, RNN, GAN을 "옛날 기술"이라며 건너뛰면 전체 발전 흐름이 끊길 수 있다.

## 워크플로우 3. LLM과 RAG를 이해하는 흐름

### 목적
트랜스포머, LLM, 프롬프팅, RAG, 에이전트의 차이를 이해한다.

### 시작 조건
- 생성형 AI가 궁금하다.
- 챗봇, RAG, 에이전트가 자꾸 헷갈린다.

### 입력
- `docs/05_transformer_and_llm.md`
- `docs/06_prompting_rag_agents.md`
- `docs/07_finetuning_vs_rag.md`

### 처리 과정
1. `05`에서 LLM이 어떻게 돌아가는지 본다.
2. `06`에서 프롬프팅, RAG, 에이전트 차이를 본다.
3. `07`에서 RAG와 파인튜닝을 비교한다.

### 결과
- "모델", "입력 설계", "외부 지식 연결", "도구 사용 시스템"이 분리되어 보인다.

### 어디서 문제가 날 수 있는가
- LLM을 데이터베이스처럼 생각하면 오해가 생긴다.
- RAG와 파인튜닝을 같은 문제 해결책으로 보면 헷갈린다.

## 워크플로우 4. 예제 코드로 감각 잡기

### 목적
문서로 본 개념을 짧은 코드 흐름으로 확인한다.

### 시작 조건
- 문서만 읽으니 감이 덜 온다.
- API 호출이 대충 어떻게 생겼는지 보고 싶다.

### 입력
- `examples/README.md`
- `examples/기초 텍스트 응답-01.py`
- `examples/구조화된 출력-02.py`
- `examples/이미지 이해-03.py`
- `examples/함수 호출-04.py`

### 처리 과정
1. `examples/README.md`로 예제 순서를 본다.
2. 가장 쉬운 텍스트 응답 예제부터 읽는다.
3. 구조화 출력, 이미지 입력, 함수 호출 순서로 올라간다.
4. 코드 주석을 따라 한 줄씩 읽는다.

### 결과
- OpenAI API 예제의 기본 형태가 보인다.
- 문서 속 개념이 실제 코드에서 어떻게 쓰이는지 감이 잡힌다.

### 어디서 문제가 날 수 있는가
- `OPENAI_API_KEY`가 없으면 실제 실행은 어렵다.
- 이 예제들은 개념 설명용이지, 완성된 서비스 코드는 아니다.

## 워크플로우 5. Python 함수 읽기 연습 흐름

### 목적
AI 코드에서 자주 보이는 함수 형태를 익힌다.

### 시작 조건
- Python 문법은 조금 아는데, AI 코드가 낯설다.

### 입력
- `ai-python-study/README.md`
- `01_math_and_data_helpers.py`
- `02_training_and_metrics_helpers.py`
- `03_nlp_and_rag_helpers.py`
- `04_llm_prompt_helpers.py`

### 처리 과정
1. `README.md`에서 파일 역할을 본다.
2. 수학/데이터 처리 함수부터 읽는다.
3. 학습/평가 함수로 넘어간다.
4. NLP/RAG, 프롬프트 헬퍼로 확장한다.

### 결과
- AI Python 코드에서 자주 보는 함수 패턴이 익숙해진다.

### 어디서 문제가 날 수 있는가
- 실무 코드와 똑같다고 기대하면 안 된다.
- 학습용 단순화가 들어가 있으므로, 실제 라이브러리와는 차이가 있다.

## 워크플로우 6. NotebookLM 업로드 후 활용 흐름

### 목적
이 레포를 NotebookLM에 올려 쉬운 요약, 질문 응답, 발표 자료 생성을 한다.

### 시작 조건
- 문서를 읽었거나 읽을 준비가 되어 있다.
- 질문하면서 공부하고 싶다.

### 입력
- `notebooklm/` 폴더 전체

### 처리 과정
1. `notebooklm/README.md`를 먼저 읽는다.
2. `notebooklm/` 문서들을 NotebookLM에 업로드한다.
3. 먼저 "이 레포를 완전 처음 보는 사람 기준으로 설명해줘." 같은 쉬운 질문부터 던진다.
4. 그다음 "핵심 개념만 설명해줘", "5장 발표 슬라이드를 만들어줘"처럼 확장한다.

### 결과
- 초보자 질문 응답
- 쉬운 요약
- 복습 자료
- 슬라이드 초안
- 발표 개요

### 어디서 문제가 날 수 있는가
- 원본 `docs/`만 넣으면 초보자에게 설명이 조금 빡빡할 수 있다.
- 그래서 `notebooklm/` 문서를 함께 또는 우선 넣는 편이 좋다.

## 관련 문서
- [04_how_the_system_works.md](04_how_the_system_works.md)
- [07_how_to_run_or_use_it.md](07_how_to_run_or_use_it.md)
- [README.md](README.md)

## 핵심 포인트
- 이 레포는 "한 번에 전부 읽기"보다 "상황별 흐름"으로 접근하는 것이 좋습니다.
- 처음 공부, 신경망 기초, LLM 이해, 예제 코드, NotebookLM 활용은 서로 다른 워크플로우입니다.
- 초보자는 `README → docs → examples/ai-python-study → notebooklm 활용` 순서가 가장 안전합니다.

## 복습 질문
- 처음 공부 시작 워크플로우에서는 어떤 문서 3개가 핵심인가?
- 신경망 기초를 잡을 때는 어떤 문서 묶음을 보면 되는가?
- NotebookLM 활용 워크플로우에서는 왜 `notebooklm/` 폴더가 중요한가?

## 다음에 읽으면 좋은 문서
- [06_important_components.md](06_important_components.md)

