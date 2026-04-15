# 이 파일은 OpenAI Responses API에서 구조화된 JSON 출력을 받는 방법을 익히기 위한 예제 코드다.

# 이 줄은 OpenAI SDK의 기본 클라이언트를 가져온다.
from openai import OpenAI

# 이 줄은 JSON 형식 문자열을 예쁘게 출력하기 위해 json 모듈을 가져온다.
import json

# 이 줄은 API 호출을 담당할 클라이언트 객체를 만든다.
client = OpenAI()

# 이 줄은 모델에게 "정해진 JSON 구조"로 답하게 만드는 요청을 시작한다.
response = client.responses.create(
    # 이 줄은 응답을 생성할 모델을 지정한다.
    model="gpt-4.1-mini",
    # 이 줄은 모델이 참고할 사용자 질문이다.
    input="RAG가 무엇인지 초보자 기준으로 정리해 줘.",
    # 이 줄은 텍스트 응답 형식을 plain text 가 아니라 JSON schema 로 강제한다.
    text={
        # 이 줄은 응답 포맷의 구체적인 종류를 JSON schema 로 지정한다.
        "format": {
            # 이 줄은 구조화 출력 기능을 쓰겠다는 의미다.
            "type": "json_schema",
            # 이 줄은 이 응답 형식의 이름을 붙이는 부분이다.
            "name": "rag_summary",
            # 이 줄은 모델이 왜 이 형식으로 답해야 하는지 설명한다.
            "description": "RAG 개념을 초보자용으로 짧고 구조적으로 정리한다.",
            # 이 줄은 모델이 따라야 할 JSON 스키마 본문이다.
            "schema": {
                # 이 줄은 최상위 JSON 타입이 객체라는 뜻이다.
                "type": "object",
                # 이 줄은 객체 안에 들어갈 필드들을 정의한다.
                "properties": {
                    # 이 줄은 concept 필드가 문자열이어야 한다는 뜻이다.
                    "concept": {"type": "string"},
                    # 이 줄은 why_it_matters 필드가 문자열이어야 한다는 뜻이다.
                    "why_it_matters": {"type": "string"},
                    # 이 줄은 when_to_use 필드가 문자열 배열이어야 한다는 뜻이다.
                    "when_to_use": {
                        # 이 줄은 when_to_use 필드의 타입을 array 로 지정한다.
                        "type": "array",
                        # 이 줄은 배열 안 각 항목의 타입을 string 으로 지정한다.
                        "items": {"type": "string"},
                        # 이 줄은 when_to_use 배열 필드 정의 블록을 닫는다.
                    },
                    # 이 줄은 properties 딕셔너리 정의를 마무리한다.
                },
                # 이 줄은 반드시 들어와야 하는 필드를 지정한다.
                "required": ["concept", "why_it_matters", "when_to_use"],
                # 이 줄은 정의하지 않은 다른 필드는 허용하지 않겠다는 뜻이다.
                "additionalProperties": False,
                # 이 줄은 schema 객체 정의를 닫는다.
            },
            # 이 줄은 모델이 스키마를 더 엄격하게 따르도록 하는 옵션이다.
            "strict": True,
            # 이 줄은 format 설정 블록을 닫는다.
        }
        # 이 줄은 text 인자에 전달하는 딕셔너리를 닫는다.
    },
    # 이 줄은 client.responses.create 함수 호출 괄호를 닫는다.
)

# 이 줄은 모델이 반환한 JSON 텍스트를 파이썬 딕셔너리로 바꾼다.
parsed_json = json.loads(response.output_text)

# 이 줄은 파싱된 결과를 사람이 읽기 쉽게 예쁘게 출력한다.
print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
