# 2026-04-08 작업 대화 익스포트

## 문서 목적
이 파일은 현재 세션에서 오간 주요 대화를 작업 단위로 정리한 내보내기본입니다.  
다른 컴퓨터에서 이어서 작업할 때, 무엇을 요청했고 무엇이 생성되었는지 빠르게 파악할 수 있도록 만들었습니다.

## 세션 요약
- AI 입문 학습용 레포 구조 설계
- `docs/` 중심 학습 문서 작성
- 프로젝트 경험 정리 문서 추가
- OpenAI Python 예제 코드 추가
- 알고리즘 정리 문서 추가
- 노션용 AI 용어집 추가
- 커리큘럼 이미지 기반으로 문서 보강
- `ai-python-study/` 학습용 코드 폴더 추가
- `notebooklm/` 초보자용 재구성 문서 세트 추가
- 정식 커리큘럼 문서 추가

---

## Turn 1

### 사용자 요청
AI 공부와 개념 잡기를 위한 개인 학습 저장소를 만들고 싶다고 요청했다.  
다음 요구가 포함됐다.
- AI 입문자가 전체 지도를 잡을 수 있어야 함
- ML, DL, LLM, RAG, Agent 개념을 체계적으로 정리
- Markdown 중심
- 한국어 작성
- 문서 끝에 다음에 읽을 문서 추천
- `README.md`와 `docs/00~11` 문서를 실제 초안으로 작성

### 어시스턴트 작업
- 레포 구조를 설계했다.
- `README.md`와 `docs/00~11` 문서를 생성했다.

### 결과
생성된 주요 파일:
- `README.md`
- `docs/00_AI_roadmap.md`
- `docs/01_AI_vs_ML_vs_DL.md`
- `docs/02_math_for_ai.md`
- `docs/03_machine_learning_basics.md`
- `docs/04_deep_learning_basics.md`
- `docs/05_transformer_and_llm.md`
- `docs/06_prompting_rag_agents.md`
- `docs/07_finetuning_vs_rag.md`
- `docs/08_evaluation_and_metrics.md`
- `docs/09_ai_system_design_basics.md`
- `docs/10_glossary.md`
- `docs/11_study_plan.md`

### 핵심 성격
- 문서 중심 학습 저장소
- 초보자 친화적 설명
- 개념 간 관계를 강조

---

## Turn 2

### 사용자 요청
지금까지 진행했던 프로젝트를 기반으로 `what_i_did.md` 같은 문서를 만들어 달라고 요청했다.  
포함하고 싶은 프로젝트 예시는 다음과 같았다.
- SDXL용 LoRA 로컬 제작
- LLM 파인튜닝 / LoRA 학습
- RAG 기반 NPC 챗봇
- 소설 축약형 RAG 문서 로컬 제작
- SLM + TTS + RAG 구현
- 데이터셋 생성 API
- 프롬프트 최적화
- 강화학습 통한 몬스터 AI 구축
- 일본어 감정분석기 로컬 제작

### 어시스턴트 작업
- 문서 이름을 더 자연스럽게 바꿔 `docs/12_projects_i_worked_on.md`로 생성했다.
- 프로젝트를 단순 나열이 아니라 기술 영역, 의미, 강점 중심으로 정리했다.
- 로컬 레포 기준으로 확인 가능한 원격 저장소 흔적도 메모했다.

### 결과
생성된 파일:
- `docs/12_projects_i_worked_on.md`

### 문서 특징
- 프로젝트별 설명
- 다뤄 본 기술 영역 표
- 강점 요약
- 클라우드 작업 흔적은 `문서 근거 부족`에 가깝게 보수적으로 정리

---

## Turn 3

### 사용자 요청
OpenAI 문서를 참고해서 Python 예제 코드를 만들고, `example` 폴더를 만들어 넣어 달라고 요청했다.  
또한 코드 줄마다 역할을 자세히 주석으로 달아 달라고 요청했다.  
파일 이름은 `"role의 요약-01"`처럼 쉬운 순서대로 붙여 달라고 했다.

### 어시스턴트 작업
- OpenAI 공식 문서 기준으로 예제 주제를 정했다.
- `examples/` 폴더를 만들고 예제와 설명용 `README.md`를 생성했다.
- 각 줄마다 촘촘한 주석을 단 Python 파일을 추가했다.

### 결과
생성된 파일:
- `examples/README.md`
- `examples/기초 텍스트 응답-01.py`
- `examples/구조화된 출력-02.py`
- `examples/이미지 이해-03.py`
- `examples/함수 호출-04.py`

### 예제 주제
- 기본 텍스트 응답
- 구조화된 출력
- 이미지 이해
- 함수 호출

---

## Turn 4

### 사용자 요청
LLM이나 RL 관련 알고리즘도 정리해 달라고 요청했다.  
예시로 `PPO`, `CQL`, `Adam` 등을 언급했다.

### 어시스턴트 작업
- 알고리즘을 단순 정의가 아니라 "어디에 쓰는지 / 왜 필요한지 / 뭐가 다른지" 중심으로 정리한 문서를 추가했다.

### 결과
생성된 파일:
- `docs/13_ai_learning_algorithms.md`

### 포함 개념
- Gradient Descent
- SGD
- Momentum
- Adam
- AdamW
- DQN
- PPO
- CQL

---

## Turn 5

### 사용자 요청
AI 용어집도 노션에 붙여 넣기 좋은 표 형식으로 만들어 달라고 요청했다.  
`하네스 엔지니어링`, `오픈클로` 같은 새로운 용어도 넣고 싶다고 했다.

### 어시스턴트 작업
- 표 중심 노션용 문서를 새로 만들었다.
- `오픈클로`는 `오픈 웨이트 / 클로즈드 모델` 맥락으로 해석해 반영했다.
- 최신 실무 용어도 함께 추가했다.

### 결과
생성된 파일:
- `docs/AI_용어집_노션용.md`

### 포함한 대표 용어
- Harness Engineering
- Context Engineering
- MCP
- Open-weight
- Closed model
- Prompt Caching
- Speculative Decoding

---

## Turn 6

### 사용자 요청
이미지로 보여 준 커리큘럼을 참고해서 기존 `docs/`를 업데이트해 달라고 요청했다.  
주요 키워드는 다음과 같았다.
- CNN, RNN, GAN
- 지도학습 / 비지도학습 / 자기지도학습 / 강화학습
- 인공신경망, weight, bias
- 선형 회귀
- Gradient Descent, GD vs SGD
- MLP, non-linear activation
- Backpropagation
- 이진 분류, 다중 분류
- Sigmoid
- likelihood, MLE
- Universal Approximation Theorem
- 깊은 신경망의 한계

### 어시스턴트 작업
- 기존 문서를 업데이트했다.
- 신경망 학습 세부 설명을 위한 신규 문서 2개를 추가했다.

### 결과
수정된 파일:
- `README.md`
- `docs/03_machine_learning_basics.md`
- `docs/04_deep_learning_basics.md`
- `docs/11_study_plan.md`
- `docs/13_ai_learning_algorithms.md`

추가된 파일:
- `docs/14_neural_network_training_and_classification.md`
- `docs/15_classic_deep_learning_architectures.md`

### 반영 내용
- 자기지도학습 추가
- GD vs SGD 설명 강화
- weight / bias / MLP / Backpropagation 보강
- CNN / RNN / GAN / UAT / 깊은 신경망 한계 정리

---

## Turn 7

### 사용자 요청
AI 관련 Python 코드에서 자주 나오는 함수를 학습할 수 있도록 `ai-python study` 폴더를 만들어 달라고 요청했다.  
주석은 풍부해야 하고, 읽으면서 이해할 수 있어야 한다고 했다.

### 어시스턴트 작업
- 폴더 이름은 경로 호환성을 위해 `ai-python-study/`로 정했다.
- 주제별 학습용 코드 파일을 만들었다.

### 결과
생성된 파일:
- `ai-python-study/README.md`
- `ai-python-study/01_math_and_data_helpers.py`
- `ai-python-study/02_training_and_metrics_helpers.py`
- `ai-python-study/03_nlp_and_rag_helpers.py`
- `ai-python-study/04_llm_prompt_helpers.py`

### 내용 예시
- seed 고정
- train/validation 분할
- sigmoid / softmax
- binary cross entropy
- precision / recall / F1
- chunking
- vocabulary
- prompt message 생성
- RAG prompt 생성

---

## Turn 8

### 사용자 요청
초보자 이해를 최우선으로 해서, NotebookLM에 올리기 좋은 별도 폴더를 만들어 달라고 요청했다.  
`docs/`를 다시 읽고, 초보자용으로 재작성된 Markdown 세트를 `notebooklm/`에 넣어 달라고 했다.

### 어시스턴트 작업
- 레포 성격을 "실행형 앱"이 아니라 "학습용 문서 저장소"로 해석했다.
- 초보자용 독립 문서 세트를 새로 작성했다.

### 결과
생성된 파일:
- `notebooklm/00_index.md`
- `notebooklm/01_this_repo_in_plain_language.md`
- `notebooklm/02_repo_structure_for_beginners.md`
- `notebooklm/03_key_concepts.md`
- `notebooklm/04_how_the_system_works.md`
- `notebooklm/05_main_workflows_step_by_step.md`
- `notebooklm/06_important_components.md`
- `notebooklm/07_how_to_run_or_use_it.md`
- `notebooklm/08_common_questions.md`
- `notebooklm/09_glossary_for_beginners.md`
- `notebooklm/10_study_questions.md`
- `notebooklm/README.md`

### 특징
- 더 쉬운 설명
- 더 작은 문서 단위
- NotebookLM Q&A, 요약, 슬라이드 생성에 최적화
- `문서 근거 부족`, `불명확`, `추정` 표기 규칙 반영

---

## Turn 9

### 사용자 요청
`docs/` 안에 정식 커리큘럼 문서도 하나 만들어 달라고 요청했다.

### 어시스턴트 작업
- `study_plan`과는 다른 역할의 "모듈형 교육 과정표" 문서를 만들었다.
- 루트 `README.md`에도 연결을 추가했다.

### 결과
생성된 파일:
- `docs/16_ai_curriculum.md`

수정된 파일:
- `README.md`

### 커리큘럼 구성
- 모듈 1: AI 전체 지도
- 모듈 2: 수학과 머신러닝 기초
- 모듈 3: 딥러닝과 신경망 기초
- 모듈 4: 딥러닝 구조와 한계
- 모듈 5: LLM과 생성형 AI
- 모듈 6: 평가와 시스템 설계
- 모듈 7: 용어, 복습, 프로젝트 연결
- 모듈 8: 예제와 Python 읽기

---

## Turn 10

### 사용자 요청
다른 컴퓨터에서도 쓸 수 있게 지금까지의 대화를 `conversation/` 폴더로 익스포트해 달라고 요청했다.

### 어시스턴트 작업
- 이 세션의 사용자-visible 작업 대화를 Markdown으로 정리했다.
- 시스템 내부 지시나 개발자 전용 설정은 제외했다.

### 결과
생성된 파일:
- `conversation/README.md`
- `conversation/2026-04-08_session_export.md`

---

## 최종 생성/수정 요약

### 새로 생성된 주요 폴더
- `docs/`
- `examples/`
- `ai-python-study/`
- `notebooklm/`
- `conversation/`

### 새로 추가된 주요 문서 번호
- `docs/12_projects_i_worked_on.md`
- `docs/13_ai_learning_algorithms.md`
- `docs/14_neural_network_training_and_classification.md`
- `docs/15_classic_deep_learning_architectures.md`
- `docs/16_ai_curriculum.md`

### 특별한 목적의 보조 문서
- `docs/AI_용어집_노션용.md`
- `notebooklm/*`
- `conversation/*`

## 현재 레포 상태 한 줄 정리
이 레포는 이제 "AI 입문 + 개념 정리 + 예제 코드 + Python 함수 학습 + NotebookLM용 초보자 자료 + 프로젝트 회고 + 커리큘럼"까지 포함한 문서 중심 학습 저장소가 되었다.

## 다음에 다른 컴퓨터에서 이어서 보면 좋은 순서
1. `conversation/2026-04-08_session_export.md`
2. `README.md`
3. `docs/16_ai_curriculum.md`
4. `docs/11_study_plan.md`
5. `notebooklm/README.md`

