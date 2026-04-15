# 이 파일은 OpenAI Vision 입력으로 이미지를 이해시키는 방법을 익히기 위한 예제 코드다.

# 이 줄은 OpenAI SDK에서 클라이언트 클래스를 가져온다.
from openai import OpenAI

# 이 줄은 로컬 이미지 파일을 base64 문자열로 바꾸기 위해 표준 라이브러리를 가져온다.
import base64

# 이 줄은 파일 경로를 더 안전하게 다루기 위해 Path 클래스를 가져온다.
from pathlib import Path

# 이 줄은 OpenAI API 클라이언트를 만든다.
client = OpenAI()

# 이 줄은 분석할 이미지 파일 경로를 지정한다.
# 실제 실행할 때는 본인 컴퓨터의 이미지 경로로 바꿔야 한다.
image_path = Path("sample_image.png")

# 이 줄은 이미지 파일을 바이너리 읽기 모드로 연다.
with image_path.open("rb") as image_file:
    # 이 줄은 이미지 바이트를 읽어서 base64 문자열로 바꾼다.
    # API에 data URL 형태로 넣기 위해 필요한 과정이다.
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

# 이 줄은 텍스트와 이미지를 함께 입력으로 보내는 응답 요청이다.
response = client.responses.create(
    # 이 줄은 이미지 입력도 이해할 수 있는 모델을 지정한다.
    model="gpt-4.1-mini",
    # 이 줄은 단순 문자열이 아니라 "메시지 배열" 형식 입력을 사용한다.
    input=[
        # 이 줄은 사용자 메시지를 담을 첫 번째 딕셔너리 블록을 연다.
        {
            # 이 줄은 이 메시지가 사용자 메시지라는 뜻이다.
            "role": "user",
            # 이 줄은 하나의 메시지 안에 여러 입력 조각을 담는 배열이다.
            "content": [
                # 이 줄은 텍스트 입력 조각 딕셔너리 블록을 연다.
                {
                    # 이 줄은 텍스트 질문 조각을 뜻한다.
                    "type": "input_text",
                    # 이 줄은 이미지에 대해 모델에게 물어볼 질문이다.
                    "text": "이 이미지에서 보이는 핵심 요소를 3가지로 요약해 줘.",
                    # 이 줄은 input_text 조각 딕셔너리 블록을 닫는다.
                },
                # 이 줄은 이미지 입력 조각 딕셔너리 블록을 새로 연다.
                {
                    # 이 줄은 이미지 입력 조각을 뜻한다.
                    "type": "input_image",
                    # 이 줄은 base64 데이터 URL 형식으로 이미지를 전달한다.
                    "image_url": f"data:image/png;base64,{image_base64}",
                    # 이 줄은 input_image 조각 딕셔너리 블록을 닫는다.
                },
                # 이 줄은 content 배열 정의를 닫는다.
            ],
            # 이 줄은 user 메시지 딕셔너리 정의를 닫는다.
        }
        # 이 줄은 input 리스트 정의를 닫는다.
    ],
    # 이 줄은 client.responses.create 함수 호출을 닫는다.
)

# 이 줄은 모델이 이미지와 질문을 바탕으로 만든 텍스트 답변을 출력한다.
print(response.output_text)
