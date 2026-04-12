# AI Python Study

이 폴더는 AI 관련 Python 코드에서 자주 보이는 함수를 "읽으면서 이해하는" 용도로 정리한 공간입니다.

## 이 폴더의 목표
- AI 코드에서 반복해서 등장하는 함수 패턴을 익힌다.
- 단순히 복붙하는 것이 아니라, 왜 이런 함수가 필요한지 이해한다.
- 풍부한 주석을 따라가며 함수의 입력, 출력, 내부 동작을 함께 본다.

## 파일 읽는 순서
1. `01_math_and_data_helpers.py`
2. `02_training_and_metrics_helpers.py`
3. `03_nlp_and_rag_helpers.py`
4. `04_llm_prompt_helpers.py`
5. `05_reinforcement_learning_helpers.py`

## 파일별 역할

| 파일 | 무엇을 담았는가 |
| --- | --- |
| `01_math_and_data_helpers.py` | seed 고정, 데이터 분할, 배치 생성, sigmoid, softmax, cosine similarity 같은 기초 함수 |
| `02_training_and_metrics_helpers.py` | loss, accuracy, precision, recall, F1, early stopping 같은 학습/평가 함수 |
| `03_nlp_and_rag_helpers.py` | 텍스트 정규화, 토큰화, chunking, vocabulary, bag-of-words, 간단 검색 함수 |
| `04_llm_prompt_helpers.py` | messages 만들기, few-shot 예시 포맷팅, RAG 프롬프트 생성, 간단 JSON 추출 함수 |
| `05_reinforcement_learning_helpers.py` | discounted return, advantage, PPO clip objective, CQL conservative penalty 같은 RL 기초 예시 |

## 추천 학습 방법
- 먼저 함수 이름만 훑지 말고 주석을 따라 한 줄씩 읽는다.
- 각 함수에서 "입력", "중간 처리", "출력"을 직접 말로 설명해 본다.
- 같은 이름의 함수가 실제 라이브러리에서는 어떻게 더 고도화되는지 비교해 본다.
- 필요하면 함수 아래 예시를 직접 실행해 본다.

## 주의할 점
- 이 폴더의 함수들은 "학습용"이라서 이해하기 쉽게 단순화한 부분이 있다.
- 실무에서는 `numpy`, `pandas`, `torch`, `scikit-learn`, `transformers` 같은 라이브러리를 함께 쓰는 경우가 많다.
- 여기서는 개념이 보이도록 최대한 표준 라이브러리와 단순 구현을 우선했다.
