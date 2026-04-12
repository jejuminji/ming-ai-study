<div align="center">

# AI Study

**신경망 기초부터 생성형 AI까지, 연결해서 이해하는 학습 저장소**

<img src="assets/min-long-long.gif" width="600" />

[![AI](https://img.shields.io/badge/AI-Fundamentals-FF6F61?style=for-the-badge&logo=openai&logoColor=white)](#-ai-fundamentals)
[![ML](https://img.shields.io/badge/ML-Machine_Learning-4ECDC4?style=for-the-badge&logo=tensorflow&logoColor=white)](#-machine-learning)
[![DL](https://img.shields.io/badge/DL-Deep_Learning-6C5CE7?style=for-the-badge&logo=pytorch&logoColor=white)](#-deep-learning)
[![LLM](https://img.shields.io/badge/LLM-Large_Language_Model-F9CA24?style=for-the-badge&logo=openai&logoColor=black)](#-llm--generative-ai)
[![RL](https://img.shields.io/badge/RL-Reinforcement_Learning-E17055?style=for-the-badge&logo=probot&logoColor=white)](#-reinforcement-learning)

---

*"개념을 따로따로 외우지 말고, 연결해서 이해하자"*

</div>

<br>

## About

AI를 처음 공부할 때 가장 어려운 건 개념이 너무 많고 서로 어떻게 연결되는지 감이 안 잡힌다는 것.  
이 레포는 **"정의"보다 "관계"** 중심으로 개념을 정리하는 개인 학습 저장소입니다.

> **대상** : AI 입문자, 개념 정리가 필요한 학습자  
> **방식** : Markdown 문서 + 주석 코드, 코드보다 개념 이해 우선  
> **목표** : 신경망 기초 ~ 생성형 AI까지 이어지는 학습 베이스 구축

<br>

## Folder Structure

```
ai-study/
├── docs/                  # 핵심 개념 문서 (주제별 번호순)
├── examples/              # OpenAI Python API 예제
├── ai-python-study/       # AI Python 함수 학습용 주석 코드
├── notebooklm/            # NotebookLM 활용 정리
├── conversation/          # 학습 대화 기록
├── video/                 # 영상 자료
└── assets/                # GIF, 이미지 등 미디어
```

> 앞으로 주제별 폴더(`rl/`, `llm/`, `dl/` 등)를 추가해 나갈 예정입니다.

<br>

## Topics

### AI Fundamentals

| # | 문서 | 핵심 내용 |
|:-:|------|-----------|
| 00 | [AI 로드맵](docs/00_AI_roadmap.md) | 전체 지도 한눈에 보기 |
| 01 | [AI vs ML vs DL](docs/01_AI_vs_ML_vs_DL.md) | 세 개념의 차이와 포함 관계 |
| 02 | [수학 기초](docs/02_math_for_ai.md) | AI에 필요한 수학의 역할 |

### Machine Learning

| # | 문서 | 핵심 내용 |
|:-:|------|-----------|
| 03 | [ML 기초](docs/03_machine_learning_basics.md) | 지도/비지도/자기지도/강화학습 |
| 13 | [학습 알고리즘](docs/13_ai_learning_algorithms.md) | GD, SGD, Adam, PPO, CQL |

### Deep Learning

| # | 문서 | 핵심 내용 |
|:-:|------|-----------|
| 04 | [딥러닝 기초](docs/04_deep_learning_basics.md) | 신경망과 딥러닝 전체 개요 |
| 14 | [신경망 훈련 & 분류](docs/14_neural_network_training_and_classification.md) | weight, bias, backprop, MLE |
| 15 | [CNN / RNN / GAN](docs/15_classic_deep_learning_architectures.md) | 고전 딥러닝 아키텍처 |

### LLM & Generative AI

| # | 문서 | 핵심 내용 |
|:-:|------|-----------|
| 05 | [Transformer & LLM](docs/05_transformer_and_llm.md) | 트랜스포머 구조와 LLM |
| 06 | [Prompting / RAG / Agents](docs/06_prompting_rag_agents.md) | 프롬프팅, RAG, 에이전트 |
| 07 | [Fine-tuning vs RAG](docs/07_finetuning_vs_rag.md) | 두 접근법의 차이 |

### Reinforcement Learning

> 강화학습 심화 문서는 준비 중입니다. 현재 관련 내용은 아래에서 확인할 수 있습니다.

| # | 문서 | 핵심 내용 |
|:-:|------|-----------|
| 03 | [ML 기초](docs/03_machine_learning_basics.md) | 강화학습 개요 (지도학습과의 비교) |
| 13 | [학습 알고리즘](docs/13_ai_learning_algorithms.md) | PPO, CQL 등 RL 알고리즘 |

### System & Practice

| # | 문서 | 핵심 내용 |
|:-:|------|-----------|
| 08 | [평가 & 메트릭](docs/08_evaluation_and_metrics.md) | 모델 평가 지표 |
| 09 | [AI 시스템 설계](docs/09_ai_system_design_basics.md) | 실무 시스템 관점 |
| 10 | [용어집](docs/10_glossary.md) | 학습용 핵심 용어 |
| 11 | [학습 계획](docs/11_study_plan.md) | 실행형 스터디 플랜 |
| 12 | [프로젝트 정리](docs/12_projects_i_worked_on.md) | 개인 프로젝트 기록 |
| 16 | [커리큘럼](docs/16_ai_curriculum.md) | 모듈형 학습 커리큘럼 |

<br>

## Learning Routes

상황에 맞는 루트를 골라 순서대로 읽어보세요.

```
 완전 입문        00 -> 01 -> 02 -> 03 -> 04 -> 14
                  AI 전체 지도 잡기 + 신경망 학습 구조 이해

 딥러닝 기초      03 -> 04 -> 13 -> 14 -> 15
                  경사하강법, MLP, 역전파, CNN/RNN/GAN

 LLM 중심        00 -> 01 -> 04 -> 13 -> 14 -> 05 -> 06 -> 07 -> 08
                  신경망 기초 위에서 Transformer, RAG, Fine-tuning

 실무 감각        00 -> 05 -> 06 -> 07 -> 08 -> 09 -> 12
                  시스템과 프로젝트 경험 관점
```

<br>

## Code Examples

<details>
<summary><b>OpenAI Python API 예제</b></summary>

| 파일 | 내용 |
|------|------|
| [기초 텍스트 응답](examples/기초%20텍스트%20응답-01.py) | 기본 Chat Completion |
| [구조화된 출력](examples/구조화된%20출력-02.py) | Structured Output |
| [이미지 이해](examples/이미지%20이해-03.py) | Vision API |
| [함수 호출](examples/함수%20호출-04.py) | Function Calling |

</details>

<details>
<summary><b>AI Python 함수 학습</b></summary>

| 파일 | 내용 |
|------|------|
| [수학 & 데이터 헬퍼](ai-python-study/01_math_and_data_helpers.py) | NumPy, 통계 기초 |
| [훈련 & 메트릭 헬퍼](ai-python-study/02_training_and_metrics_helpers.py) | 학습 루프, 평가 |
| [NLP & RAG 헬퍼](ai-python-study/03_nlp_and_rag_helpers.py) | 텍스트 처리, 검색 |
| [LLM 프롬프트 헬퍼](ai-python-study/04_llm_prompt_helpers.py) | 프롬프트 엔지니어링 |
| [강화학습 헬퍼](ai-python-study/05_reinforcement_learning_helpers.py) | PPO, CQL, return, advantage |

</details>

<br>

## Tips

- 처음엔 "완벽히 이해"보다 **"지도 익히기"** 에 집중하세요
- 문서를 읽다 막히면 [용어집](docs/10_glossary.md)을 참고하세요
- 한 번 읽고 끝내지 말고, 프로젝트 하면서 **반복 참조**하세요
- 새로 배운 내용은 같은 형식으로 계속 추가해서 나만의 지식 베이스로 키우세요

<br>

---

<div align="center">

*계속 업데이트 중입니다*

</div>
