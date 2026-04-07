# 이 줄은 OpenAI SDK의 클라이언트 클래스를 가져온다.
from openai import OpenAI

# 이 줄은 모델이 넘겨준 함수 인자를 JSON으로 읽기 위해 json 모듈을 가져온다.
import json

# 이 줄은 OpenAI API 클라이언트를 만든다.
client = OpenAI()


# 이 줄은 실제 애플리케이션 쪽에서 실행할 날씨 조회 함수를 예시로 만든 것이다.
def get_weather(location: str, unit: str) -> str:
    # 이 줄은 예제를 단순하게 유지하기 위해 하드코딩된 결과를 반환한다.
    fake_weather_data = {
        "Seoul": {"celsius": "18도, 맑음", "fahrenheit": "64F, sunny"},
        "Tokyo": {"celsius": "20도, 흐림", "fahrenheit": "68F, cloudy"},
    }
    # 이 줄은 location 이 사전에 있으면 해당 값을 주고, 없으면 기본 문구를 준다.
    return fake_weather_data.get(location, {}).get(unit, "날씨 정보를 찾지 못했습니다.")


# 이 줄은 모델에게 보여 줄 함수 도구 정의 목록이다.
tools = [
    {
        # 이 줄은 이 도구가 함수형 도구라는 뜻이다.
        "type": "function",
        # 이 줄은 모델이 호출할 함수 이름이다.
        "name": "get_weather",
        # 이 줄은 함수가 언제 쓰이는지 설명한다.
        "description": "주어진 도시의 현재 날씨를 조회한다.",
        # 이 줄은 함수 입력 인자의 JSON schema 를 정의한다.
        "parameters": {
            # 이 줄은 함수 인자가 객체 형식이라는 뜻이다.
            "type": "object",
            # 이 줄은 허용할 필드들을 정의한다.
            "properties": {
                # 이 줄은 location 이 문자열이어야 함을 나타낸다.
                "location": {
                    "type": "string",
                    "description": "예: Seoul 또는 Tokyo",
                },
                # 이 줄은 unit 이 문자열이며 선택 가능한 값이 정해져 있음을 나타낸다.
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "온도 단위",
                },
            },
            # 이 줄은 두 필드가 모두 필수라는 뜻이다.
            "required": ["location", "unit"],
            # 이 줄은 정의하지 않은 다른 필드는 받지 않겠다는 뜻이다.
            "additionalProperties": False,
        },
        # 이 줄은 스키마를 더 엄격하게 따르도록 하는 옵션이다.
        "strict": True,
    }
]

# 이 줄은 먼저 모델에게 질문과 함께 사용할 수 있는 도구 목록을 보낸다.
first_response = client.responses.create(
    # 이 줄은 함수 호출을 지원하는 모델을 지정한다.
    model="gpt-4.1-mini",
    # 이 줄은 사용자 질문이다.
    input="서울 날씨를 섭씨 기준으로 알려줘.",
    # 이 줄은 모델이 필요하면 호출할 수 있는 함수 목록이다.
    tools=tools,
)

# 이 줄은 첫 번째 응답 안에 들어 있는 출력 항목들을 순회한다.
for item in first_response.output:
    # 이 줄은 함수 호출 항목이 아니면 건너뛴다.
    if item.type != "function_call":
        continue

    # 이 줄은 모델이 함수에 넘기려는 인자 문자열을 파이썬 딕셔너리로 바꾼다.
    arguments = json.loads(item.arguments)

    # 이 줄은 실제 파이썬 함수를 실행해서 결과를 만든다.
    tool_result = get_weather(
        location=arguments["location"],
        unit=arguments["unit"],
    )

    # 이 줄은 함수 실행 결과를 다시 모델에게 전달해서 최종 자연어 답변을 받는다.
    second_response = client.responses.create(
        # 이 줄은 같은 모델을 다시 사용한다.
        model="gpt-4.1-mini",
        # 이 줄은 이전 응답 ID를 넘겨 대화 상태를 이어간다.
        previous_response_id=first_response.id,
        # 이 줄은 함수 실행 결과를 function_call_output 형식으로 보낸다.
        input=[
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": tool_result,
            }
        ],
    )

    # 이 줄은 최종적으로 모델이 정리한 자연어 답변을 출력한다.
    print(second_response.output_text)
