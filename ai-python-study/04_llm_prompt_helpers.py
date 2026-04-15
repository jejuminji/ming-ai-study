# 이 파일은 LLM 애플리케이션 코드에서 자주 보이는 프롬프트/메시지 헬퍼를 모아 둔 학습용 파일이다.

# JSON 문자열을 파싱하기 위해 json 모듈을 가져온다.
import json

# 정규 표현식 기반으로 JSON 블록을 찾기 위해 re 모듈을 가져온다.
import re

# 타입 힌트를 위해 typing 도구를 가져온다.
from typing import Dict, List, Optional, Sequence


# 이 함수는 system, user, few-shot 예시를 합쳐 chat messages 리스트를 만든다.
def build_messages(
    # 이 매개변수는 대화 전체의 역할과 규칙을 담는 system prompt 문자열이다.
    system_prompt: str,
    # 이 매개변수는 실제 사용자가 보낸 질문 문자열이다.
    user_prompt: str,
    # 이 매개변수는 user/assistant 예시 대화를 선택적으로 받는다.
    examples: Optional[Sequence[Dict[str, str]]] = None,
    # 이 반환 타입 표시는 role 과 content 를 가진 메시지 딕셔너리 리스트를 돌려준다는 뜻이다.
) -> List[Dict[str, str]]:
    # messages 리스트를 system 메시지 하나로 시작한다.
    messages = [{"role": "system", "content": system_prompt}]

    # few-shot 예시가 있으면 차례대로 대화 기록처럼 추가한다.
    if examples:
        # for 반복문으로 examples 시퀀스의 각 예시 딕셔너리를 하나씩 읽는다.
        for example in examples:
            # example 에는 user 와 assistant 키가 있다고 가정한다.
            messages.append({"role": "user", "content": example["user"]})
            # assistant 예시 답변도 append 함수로 뒤에 이어 붙인다.
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
        # 첫 줄에는 Example 번호를 넣어 각 예시 블록의 시작을 표시한다.
        lines.append(f"[Example {index}]")
        # 두 번째 줄에는 user 예시 문장을 기록한다.
        lines.append(f"User: {example['user']}")
        # 세 번째 줄에는 assistant 예시 답변을 기록한다.
        lines.append(f"Assistant: {example['assistant']}")
        # 빈 문자열 한 줄을 추가해 예시 블록 사이를 띄운다.
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
        # 첫 번째 문자열은 검색 문맥만 사용하라는 핵심 지시를 담는다.
        "다음 참고 문맥만 사용해서 질문에 답해 주세요.\n"
        # 두 번째 문자열은 근거가 부족할 때는 모른다고 답하라는 안전장치다.
        "근거가 부족하면 모른다고 답해 주세요.\n\n"
        # 이 f-string 은 format_context_blocks 결과를 프롬프트 본문에 삽입한다.
        f"{context_text}\n\n"
        # 마지막 f-string 은 실제 질문 문장을 붙여 프롬프트를 완성한다.
        f"질문: {question}"
        # 이 줄은 여러 줄 문자열을 묶은 prompt 정의를 닫는다.
    )

    # 완성된 프롬프트를 반환한다.
    return prompt


# 이 함수는 텍스트 안에서 첫 번째 JSON 객체처럼 보이는 부분을 찾아 파싱한다.
def safe_extract_json_object(text: str) -> Optional[Dict]:
    # 가장 단순한 학습용 방식으로 중괄호 블록을 찾아본다.
    match = re.search(r"\{.*\}", text, re.DOTALL)

    # JSON 블록처럼 보이는 부분이 없으면 None 을 반환한다.
    if not match:
        # None 반환은 안전하게 "추출 실패"를 호출자에게 알려 주는 방식이다.
        return None

    # 찾은 블록만 따로 꺼낸다.
    json_candidate = match.group(0)

    # JSON 파싱을 시도한다.
    try:
        # json.loads 함수로 문자열 json_candidate 를 파이썬 딕셔너리로 바꾼다.
        return json.loads(json_candidate)
    # 파싱에 실패하면 None 을 반환해 호출 쪽에서 후처리하게 한다.
    except json.JSONDecodeError:
        # JSONDecodeError 가 나면 None 을 반환해 예외 대신 실패 결과를 넘긴다.
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
            # break 로 반복을 중단해 문자 수 예산을 초과하지 않게 한다.
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
        # 첫 번째 예시는 RAG 질문과 답변을 짧게 담는다.
        {"user": "RAG가 뭐야?", "assistant": "검색된 문서를 참고해 답하는 방식이야."},
        # 두 번째 예시는 LoRA 질문과 답변을 담아 few-shot 예시를 늘린다.
        {"user": "LoRA가 뭐야?", "assistant": "적은 파라미터만 학습하는 튜닝 방식이야."},
        # 이 줄은 examples 리스트 정의를 닫는다.
    ]

    # 메시지 리스트를 만든 뒤 출력한다.
    print(
        # 첫 번째 print 인자는 출력 라벨인 messages 문자열이다.
        "messages:",
        # build_messages 함수 호출로 완성한 메시지 리스트를 두 번째 인자로 넘긴다.
        build_messages(
            # system_prompt 인자에는 AI 튜터 역할 지시를 넣는다.
            system_prompt="너는 친절한 AI 튜터다.",
            # user_prompt 인자에는 PPO 설명 요청 문장을 넣는다.
            user_prompt="PPO를 쉬운 말로 설명해 줘.",
            # examples 인자에는 위에서 만든 few-shot 예시 리스트를 전달한다.
            examples=examples,
            # 이 줄은 build_messages 함수 호출 괄호를 닫는다.
        ),
        # 이 줄은 print 함수 호출 괄호를 닫는다.
    )

    # few-shot 포맷팅 결과를 출력한다.
    print("few_shot_text:")
    # format_few_shot_examples 함수가 만든 문자열을 print 함수로 보여 준다.
    print(format_few_shot_examples(examples))

    # RAG 프롬프트 예시를 출력한다.
    print(
        # 첫 번째 print 인자는 rag_prompt 라벨 문자열이다.
        "rag_prompt:",
        # make_rag_prompt 함수로 만든 최종 프롬프트 문자열을 함께 출력한다.
        make_rag_prompt(
            # question 인자에는 답변받고 싶은 질문 문자열을 넣는다.
            question="RAG의 장점은?",
            # contexts 인자에는 검색된 참고 문맥 리스트를 전달한다.
            contexts=["RAG는 외부 문서를 검색해 답변에 반영한다.", "최신 정보를 붙이기 쉽다."],
            # 이 줄은 make_rag_prompt 함수 호출을 닫는다.
        ),
        # 이 줄은 print 함수 호출 전체를 닫는다.
    )

    # JSON 추출 예시를 출력한다.
    print("json_object:", safe_extract_json_object('결과는 {"label": "positive", "score": 0.91} 입니다.'))
