# 이 파일은 OpenAI Responses API로 기본 텍스트 응답을 받는 방법을 익히기 위한 예제 코드다.

# 이 줄은 OpenAI Python SDK에서 OpenAI 클래스를 가져오는 부분이다.
from openai import OpenAI

# 이 줄은 OpenAI API와 통신할 클라이언트 객체를 만든다.
# 환경 변수 OPENAI_API_KEY 가 미리 설정되어 있으면 자동으로 그 키를 사용한다.
client = OpenAI()

# 이 줄은 Responses API를 호출해서 모델에게 텍스트 응답 생성을 요청한다.
response = client.responses.create(
    # 이 줄은 어떤 모델을 사용할지 지정한다.
    # 처음 예제를 따라 할 때는 가볍고 빠른 모델부터 써 보는 편이 부담이 적다.
    model="gpt-4.1-mini",
    # 이 줄은 모델에게 전달할 사용자 입력이다.
    input="AI, 머신러닝, 딥러닝의 차이를 초보자도 이해하게 3문장으로 설명해 줘.",
    # 이 줄은 client.responses.create 함수 호출 괄호를 닫는다.
)

# 이 줄은 모델이 만든 최종 텍스트를 보기 쉽게 출력한다.
# output_text 는 텍스트 응답만 바로 꺼내 볼 수 있게 도와주는 편한 속성이다.
print(response.output_text)
