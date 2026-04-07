# 이 파일은 LLM 애플리케이션 코드에서 자주 보이는 프롬프트/메시지 헬퍼를 모아 둔 학습용 파일이다.

# JSON 문자열을 파싱하기 위해 json 모듈을 가져온다.
import json

# 정규 표현식 기반으로 JSON 블록을 찾기 위해 re 모듈을 가져온다.
import re

# 타입 힌트를 위해 typing 도구를 가져온다.
from typing import Dict, List, Optional, Sequence


# 이 함수는 system, user, few-shot 예시를 합쳐 chat messages 리스트를 만든다.
def build_messages(
    system_prompt: str,
    user_prompt: str,
    examples: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    # messages 리스트를 system 메시지 하나로 시작한다.
    messages = [{"role": "system", "content": system_prompt}]

    # few-shot 예시가 있으면 차례대로 대화 기록처럼 추가한다.
    if examples:
        for example in examples:
            # example 에는 user 와 assistant 키가 있다고 가정한다.
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})

    # 마지막으로 실제 사용자 질문을 추가한다.
    messages.append({"role": "user", "content": user_prompt})

    # 완성된 messages 리스트를 반환한다.
    return messages


# 이 함수는 few-shot 예시를 하나의 읽기 쉬운 문자열로 포맷팅한다.
def format_few_shot_examples(examples: Sequence[Dict[str, str]]) -> str:
    # 여러 줄을 모아 최종 문자열을 만들기 위해 리스트를 준비한다.
    lines = []

    # 예시를 하나씩 순회하면서 번호와 함께 문자열로 만든다.
    for index, example in enumerate(examples, start=1):
        lines.append(f"[Example {index}]")
        lines.append(f"User: {example['user']}")
        lines.append(f"Assistant: {example['assistant']}")
        lines.append("")

    # 줄들을 줄바꿈으로 이어 붙여 반환한다.
    return "\n".join(lines).strip()


# 이 함수는 RAG 에서 검색된 문맥들을 프롬프트에 넣기 좋은 텍스트 블록으로 바꾼다.
def format_context_blocks(contexts: Sequence[str]) -> str:
    # 각 문맥 앞에 번호를 붙여 참조하기 쉽게 만든다.
    blocks = [f"[Context {index}] {context}" for index, context in enumerate(contexts, start=1)]

    # 문맥 블록을 줄바꿈으로 연결해 반환한다.
    return "\n".join(blocks)


# 이 함수는 질문과 검색 문맥을 바탕으로 RAG 프롬프트를 만든다.
def make_rag_prompt(question: str, contexts: Sequence[str]) -> str:
    # 검색 문맥을 읽기 쉬운 블록 형태로 먼저 포맷팅한다.
    context_text = format_context_blocks(contexts)

    # 모델에게 원하는 행동을 분명히 하기 위한 지시문을 만든다.
    prompt = (
        "다음 참고 문맥만 사용해서 질문에 답해 주세요.\n"
        "근거가 부족하면 모른다고 답해 주세요.\n\n"
        f"{context_text}\n\n"
        f"질문: {question}"
    )

    # 완성된 프롬프트를 반환한다.
    return prompt


# 이 함수는 텍스트 안에서 첫 번째 JSON 객체처럼 보이는 부분을 찾아 파싱한다.
def safe_extract_json_object(text: str) -> Optional[Dict]:
    # 가장 단순한 학습용 방식으로 중괄호 블록을 찾아본다.
    match = re.search(r"\{.*\}", text, re.DOTALL)

    # JSON 블록처럼 보이는 부분이 없으면 None 을 반환한다.
    if not match:
        return None

    # 찾은 블록만 따로 꺼낸다.
    json_candidate = match.group(0)

    # JSON 파싱을 시도한다.
    try:
        return json.loads(json_candidate)
    # 파싱에 실패하면 None 을 반환해 호출 쪽에서 후처리하게 한다.
    except json.JSONDecodeError:
        return None


# 이 함수는 너무 긴 문맥 리스트를 문자 수 기준으로 잘라 컨텍스트 예산을 맞춘다.
def trim_contexts_by_char_budget(contexts: Sequence[str], max_chars: int) -> List[str]:
    # 선택된 문맥을 담을 빈 리스트를 만든다.
    selected_contexts = []

    # 현재까지 사용한 문자 수를 0으로 시작한다.
    used_chars = 0

    # 문맥을 앞에서부터 순회하면서 예산 안에 들어오면 추가한다.
    for context in contexts:
        # 현재 문맥 길이를 계산한다.
        context_length = len(context)

        # 현재 문맥을 넣으면 예산을 넘는지 확인한다.
        if used_chars + context_length > max_chars:
            break

        # 예산 안이면 결과 리스트에 넣는다.
        selected_contexts.append(context)

        # 사용 문자 수를 갱신한다.
        used_chars += context_length

    # 예산 안에 들어온 문맥만 반환한다.
    return selected_contexts


# 이 아래 코드는 파일을 직접 실행했을 때 동작하는 간단한 예시다.
if __name__ == "__main__":
    # few-shot 예시 데이터를 만든다.
    examples = [
        {"user": "RAG가 뭐야?", "assistant": "검색된 문서를 참고해 답하는 방식이야."},
        {"user": "LoRA가 뭐야?", "assistant": "적은 파라미터만 학습하는 튜닝 방식이야."},
    ]

    # 메시지 리스트를 만든 뒤 출력한다.
    print(
        "messages:",
        build_messages(
            system_prompt="너는 친절한 AI 튜터다.",
            user_prompt="PPO를 쉬운 말로 설명해 줘.",
            examples=examples,
        ),
    )

    # few-shot 포맷팅 결과를 출력한다.
    print("few_shot_text:")
    print(format_few_shot_examples(examples))

    # RAG 프롬프트 예시를 출력한다.
    print(
        "rag_prompt:",
        make_rag_prompt(
            question="RAG의 장점은?",
            contexts=["RAG는 외부 문서를 검색해 답변에 반영한다.", "최신 정보를 붙이기 쉽다."],
        ),
    )

    # JSON 추출 예시를 출력한다.
    print("json_object:", safe_extract_json_object('결과는 {"label": "positive", "score": 0.91} 입니다.'))
