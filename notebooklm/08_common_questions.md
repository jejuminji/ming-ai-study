# 초보자가 자주 하는 질문

## 이 문서의 목적
이 레포를 처음 보는 사람이 자주 할 만한 질문을 모아, 짧고 쉽게 답합니다.

## 추천 독자 수준
beginner

## 아주 짧은 요약
처음 헷갈리는 질문들을 먼저 정리해 두면 레포를 훨씬 편하게 읽을 수 있습니다.

## Q1. 이 프로젝트의 핵심은 뭐야?
이 레포의 핵심은 "AI를 처음 공부할 때 길을 잃지 않게 해 주는 문서형 학습 저장소"라는 점입니다.  
특히 AI, 머신러닝, 딥러닝, LLM, RAG, 에이전트를 따로따로 외우지 않고 연결해서 이해하게 도와줍니다.

## Q2. 이건 앱이야, 문서 모음이야?
문서 모음에 더 가깝습니다.  
코드가 아예 없는 것은 아니지만, 중심은 실행형 서비스가 아니라 학습 자료입니다.

## Q3. 어디부터 읽어야 해?
가장 안전한 시작 순서는 이렇습니다.
1. `README.md`
2. `docs/00_AI_roadmap.md`
3. `docs/01_AI_vs_ML_vs_DL.md`
4. `docs/03_machine_learning_basics.md`
5. `docs/04_deep_learning_basics.md`

## Q4. 전체 흐름은 어떻게 돼?
큰 흐름은 아래처럼 보면 됩니다.
1. AI 전체 지도 보기
2. 머신러닝과 딥러닝 기초 이해
3. 신경망 학습 원리와 알고리즘 이해
4. LLM, RAG, 에이전트 이해
5. 평가와 시스템 설계 이해
6. 예제 코드와 프로젝트 경험으로 연결

## Q5. 가장 중요한 파일은 뭐야?
초보자 기준으로는 다음 파일들이 가장 중요합니다.
- `README.md`
- `docs/00_AI_roadmap.md`
- `docs/03_machine_learning_basics.md`
- `docs/04_deep_learning_basics.md`
- `docs/05_transformer_and_llm.md`
- `docs/06_prompting_rag_agents.md`

## Q6. LLM만 빨리 보면 안 돼?
볼 수는 있지만, 완전 초보자라면 신경망 기초를 먼저 보는 편이 훨씬 덜 헷갈립니다.  
그래서 `docs/04`, `docs/13`, `docs/14`를 먼저 훑고 `docs/05`로 가는 흐름이 더 안정적입니다.

## Q7. 이 레포에서 RAG와 파인튜닝은 어디서 봐?
- `docs/06_prompting_rag_agents.md`: RAG와 에이전트
- `docs/07_finetuning_vs_rag.md`: 파인튜닝과 RAG 비교

## Q8. 예제 코드는 어디 있어?
`examples/` 폴더에 있습니다.  
OpenAI Python 예제들이 쉬운 순서대로 정리되어 있습니다.

## Q9. Python 함수 공부용 자료도 있어?
있습니다. `ai-python-study/` 폴더가 그 역할을 합니다.  
AI 관련 Python 코드에서 자주 나오는 함수들을 주석과 함께 읽을 수 있게 만들어 두었습니다.

## Q10. 실행하려면 무엇이 필요해?
문서만 읽을 때는 특별한 준비가 거의 필요 없습니다.  
`examples/` 실행에는 Python 3.10+, `openai` 패키지, `OPENAI_API_KEY`가 필요합니다.

## Q11. 이 레포는 실전 감각도 있어?
어느 정도 있습니다.  
단순 개념 문서만 있는 것이 아니라, `docs/12_projects_i_worked_on.md`처럼 실제 프로젝트 경험과 연결된 문서도 있습니다.

## Q12. 이 레포 하나로 AI를 다 배울 수 있어?
아니요. 이 레포는 "길잡이"에 더 가깝습니다.  
즉, 전체 지도를 잡고, 핵심 개념을 연결해서 이해하게 도와주는 역할이 큽니다.

## Q13. NotebookLM에는 무엇을 올리면 좋아?
가장 좋은 방법은 `notebooklm/` 폴더 문서들을 먼저 올리는 것입니다.  
원본 `docs/`보다 더 쉽게 풀어 쓴 문서라서, 초보자 질문 응답과 요약에 더 잘 맞습니다.

## Q14. 이 레포에서 불명확한 부분도 있어?
있습니다.
- 하나의 통합 실행 앱 구조: 문서 근거 부족
- 모든 프로젝트를 그대로 재현하는 실행 절차: 문서 근거 부족
- 클라우드 배포 세부 흐름: 문서 근거 부족

## 관련 문서
- [01_this_repo_in_plain_language.md](01_this_repo_in_plain_language.md)
- [07_how_to_run_or_use_it.md](07_how_to_run_or_use_it.md)
- [09_glossary_for_beginners.md](09_glossary_for_beginners.md)

## 핵심 포인트
- 이 레포는 문서형 AI 학습 저장소입니다.
- 가장 먼저 봐야 할 것은 `README.md`와 `docs/00`입니다.
- 예제는 `examples/`, 함수 학습은 `ai-python-study/`가 담당합니다.

## 복습 질문
- 이 레포는 앱 레포인가, 문서형 학습 레포인가?
- 예제 코드는 어느 폴더에서 찾을 수 있는가?
- LLM만 먼저 보면 왜 헷갈릴 수 있는가?

## 다음에 읽으면 좋은 문서
- [09_glossary_for_beginners.md](09_glossary_for_beginners.md)

