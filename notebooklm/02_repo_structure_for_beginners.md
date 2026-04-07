# 초보자를 위한 레포 구조 설명

## 이 문서의 목적
이 저장소 안에 어떤 폴더와 파일이 있는지, 초보자는 어디부터 봐야 하는지 쉽게 설명합니다.

## 추천 독자 수준
beginner

## 아주 짧은 요약
처음에는 `README.md`, `docs/`, `examples/`, `ai-python-study/`만 이해해도 충분합니다.

## 먼저 전체 구조를 아주 단순하게 보기

| 위치 | 역할 | 초보자에게 쉬운 설명 |
| --- | --- | --- |
| `README.md` | 전체 안내 | 이 레포의 입구 |
| `docs/` | 핵심 공부 문서 | 이 레포의 본문 |
| `examples/` | OpenAI 예제 코드 | 따라 읽는 짧은 실습 예제 |
| `ai-python-study/` | 함수 학습 코드 | AI Python 코드 읽기 연습장 |
| `notebooklm/` | 초보자용 재구성 문서 | NotebookLM 업로드용 쉬운 요약본 |

## 꼭 먼저 볼 것

### 1. `README.md`
왜 먼저 보나: 이 레포가 무엇을 위한 곳인지, 어떤 순서로 읽어야 하는지 알려 주기 때문입니다.

### 2. `docs/00_AI_roadmap.md`
왜 먼저 보나: 전체 지도를 잡아 주기 때문입니다.

### 3. `docs/01_AI_vs_ML_vs_DL.md`
왜 먼저 보나: 가장 기본적인 구분이 여기서 잡히기 때문입니다.

### 4. `docs/03_machine_learning_basics.md`
왜 먼저 보나: 지도학습, 비지도학습, 자기지도학습, 강화학습의 차이를 여기서 잡을 수 있기 때문입니다.

### 5. `docs/04_deep_learning_basics.md`
왜 먼저 보나: 신경망, weight, bias, 활성화 함수 같은 기초 흐름이 여기서 시작되기 때문입니다.

## 나중에 봐도 되는 것

### `docs/05_transformer_and_llm.md`
LLM이 궁금할 때 매우 중요하지만, 완전 처음에는 신경망 기초 뒤에 보는 편이 더 쉽습니다.

### `docs/06_prompting_rag_agents.md`
실무에서 많이 나오지만, LLM 구조를 모르면 용어가 한꺼번에 쏟아질 수 있습니다.

### `docs/08_evaluation_and_metrics.md`
중요하지만, 모델과 시스템 개념을 어느 정도 본 뒤 읽는 편이 이해가 쉽습니다.

### `docs/09_ai_system_design_basics.md`
초반보다 중후반에 보면 더 잘 들어옵니다.

## 참고용 파일

### `docs/10_glossary.md`
용어가 막힐 때 바로 찾아보는 사전 같은 문서입니다.

### `docs/11_study_plan.md`
4주 학습 루트를 보여 주는 문서입니다.

### `docs/12_projects_i_worked_on.md`
실제 프로젝트 경험을 정리한 문서입니다. 개념을 현실과 연결할 때 좋습니다.

### `docs/AI_용어집_노션용.md`
노션에 붙여 넣기 쉽게 만든 표 중심 용어집입니다.

### `examples/`
코드 실행을 꼭 해야 하는 것은 아니지만, API 사용 감각을 보기 좋습니다.

### `ai-python-study/`
실무 함수가 아니라 학습용 함수 설명 코드입니다. 코드를 읽는 훈련에 좋습니다.

## `docs/` 안을 조금 더 쉽게 나누기

| 문서 묶음 | 포함 파일 | 무슨 내용인가 |
| --- | --- | --- |
| 전체 지도 | `00`, `01`, `02` | AI가 무엇인지, 어떤 큰 그림인지 |
| 기초 학습 | `03`, `04`, `13`, `14`, `15` | 머신러닝, 신경망, 학습 알고리즘, 분류, CNN/RNN/GAN |
| 생성형 AI | `05`, `06`, `07` | LLM, 프롬프팅, RAG, 에이전트, 파인튜닝 |
| 운영 관점 | `08`, `09` | 평가, 지표, 시스템 설계 |
| 보조 문서 | `10`, `11`, `12`, `AI_용어집_노션용` | 용어집, 계획, 프로젝트 경험 |

## 초보자가 가장 덜 헤매는 시작 순서
1. `README.md`
2. `docs/00_AI_roadmap.md`
3. `docs/01_AI_vs_ML_vs_DL.md`
4. `docs/03_machine_learning_basics.md`
5. `docs/04_deep_learning_basics.md`
6. `docs/13_ai_learning_algorithms.md`
7. `docs/14_neural_network_training_and_classification.md`

## 불명확하거나 주의할 점
- 이 레포에는 하나의 통합 앱 구조가 분명히 잡혀 있는 것은 아닙니다.
- 그래서 폴더 역할도 "코드 모듈 관계"보다 "학습 자료 묶음"으로 이해하는 편이 맞습니다.
- 통합 실행 절차는 `문서 근거 부족`입니다. 대신 예제와 학습용 코드 실행 정보는 일부 존재합니다.

## 관련 문서
- [01_this_repo_in_plain_language.md](01_this_repo_in_plain_language.md)
- [06_important_components.md](06_important_components.md)
- [07_how_to_run_or_use_it.md](07_how_to_run_or_use_it.md)

## 핵심 포인트
- 초보자는 모든 파일을 한 번에 보지 않아도 됩니다.
- 가장 중요한 곳은 `README.md`와 `docs/`입니다.
- `examples/`와 `ai-python-study/`는 이해를 돕는 보조 자료입니다.

## 복습 질문
- 이 레포에서 가장 먼저 봐야 할 파일 3개는 무엇인가?
- `docs/`와 `examples/`의 역할 차이는 무엇인가?
- `ai-python-study/`는 실무 코드 폴더인가, 학습용 코드 폴더인가?

## 다음에 읽으면 좋은 문서
- [03_key_concepts.md](03_key_concepts.md)

