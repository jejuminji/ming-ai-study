# 이 파일은 NLP 전처리와 RAG 준비 단계에서 자주 보이는 함수를 모아 둔 학습용 파일이다.

# 정규 표현식을 쓰기 위해 re 모듈을 가져온다.
import re

# Counter 는 단어 빈도 계산에 유용하므로 collections 에서 가져온다.
from collections import Counter

# 타입 힌트를 위해 typing 도구를 가져온다.
from typing import Dict, List, Sequence, Tuple


# 이 함수는 텍스트를 단순 정규화한다.
def normalize_text(text: str) -> str:
    # 앞뒤 공백을 제거해 불필요한 여백을 정리한다.
    text = text.strip()

    # 줄바꿈과 탭을 공백으로 바꿔 한 줄 처리하기 쉽게 만든다.
    text = text.replace("\n", " ").replace("\t", " ")

    # 연속된 공백은 하나로 줄인다.
    text = re.sub(r"\s+", " ", text)

    # 학습용 예시에서는 소문자 통일을 통해 단어 분산을 줄인다.
    text = text.lower()

    # 정리된 텍스트를 반환한다.
    return text


# 이 함수는 아주 단순한 토큰화를 수행한다.
def simple_tokenize(text: str) -> List[str]:
    # 먼저 텍스트를 정규화한다.
    normalized = normalize_text(text)

    # 영어, 숫자, 한글 단위 토큰만 대략 뽑아낸다.
    tokens = re.findall(r"[a-z0-9가-힣]+", normalized)

    # 추출된 토큰 리스트를 반환한다.
    return tokens


# 이 함수는 여러 문서에서 단어 빈도를 계산한다.
def word_frequency(texts: Sequence[str], top_k: int = 10) -> List[Tuple[str, int]]:
    # 모든 텍스트를 돌면서 토큰을 모으기 위한 Counter 를 만든다.
    counter = Counter()

    # 각 텍스트를 토큰화해서 Counter 에 누적한다.
    for text in texts:
        # Counter.update 함수로 simple_tokenize 결과 토큰 빈도를 더한다.
        counter.update(simple_tokenize(text))

    # 가장 많이 나온 단어 top_k 개를 반환한다.
    return counter.most_common(top_k)


# 이 함수는 vocabulary, 즉 단어장에서 단어와 인덱스 매핑을 만든다.
def build_vocabulary(texts: Sequence[str], min_freq: int = 1) -> Dict[str, int]:
    # 모든 단어 빈도를 먼저 센다.
    counter = Counter()

    # 각 텍스트를 토큰화해 빈도를 누적한다.
    for text in texts:
        # simple_tokenize 함수로 쪼갠 단어들을 Counter.update 로 합산한다.
        counter.update(simple_tokenize(text))

    # vocabulary 딕셔너리를 비어 있는 상태로 시작한다.
    vocabulary = {}

    # 빈도가 min_freq 이상인 단어만 vocabulary 에 넣는다.
    for token, frequency in counter.items():
        # 현재 단어의 빈도가 최소 기준을 만족하는지 확인한다.
        if frequency >= min_freq:
            # vocabulary 딕셔너리에 새 단어를 넣고 현재 길이를 인덱스로 사용한다.
            vocabulary[token] = len(vocabulary)

    # 완성된 vocabulary 를 반환한다.
    return vocabulary


# 이 함수는 텍스트를 bag-of-words 벡터로 바꾼다.
def text_to_bow_vector(text: str, vocabulary: Dict[str, int]) -> List[int]:
    # vocabulary 크기만큼 0으로 채운 벡터를 만든다.
    vector = [0] * len(vocabulary)

    # 텍스트를 토큰화한 뒤 각 토큰이 vocabulary 에 있으면 해당 위치 값을 1 증가시킨다.
    for token in simple_tokenize(text):
        # 현재 토큰이 vocabulary 딕셔너리에 등록된 단어인지 확인한다.
        if token in vocabulary:
            # token_index 변수에 해당 단어의 정수 인덱스를 저장한다.
            token_index = vocabulary[token]
            # vector 리스트의 해당 위치를 1 늘려 단어 출현 횟수를 반영한다.
            vector[token_index] += 1

    # 완성된 bag-of-words 벡터를 반환한다.
    return vector


# 이 함수는 긴 텍스트를 chunk_size 단위로 나누되 overlap 을 조금 남긴다.
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    # overlap 이 chunk_size 이상이면 같은 부분만 반복될 수 있으므로 예외를 발생시킨다.
    if overlap >= chunk_size:
        # ValueError 로 overlap 값이 chunk_size 보다 작아야 한다는 규칙을 알린다.
        raise ValueError("overlap must be smaller than chunk_size.")

    # 텍스트를 공백 기준으로 나눠 단어 리스트를 만든다.
    words = normalize_text(text).split(" ")

    # 결과 청크를 담을 빈 리스트를 만든다.
    chunks = []

    # 한 번 이동할 때 실제로 앞으로 나아갈 단어 수를 계산한다.
    step = chunk_size - overlap

    # 단어 리스트를 step 간격으로 순회하며 chunk 를 만든다.
    for start_index in range(0, len(words), step):
        # 현재 시작 지점부터 chunk_size 만큼 단어를 가져온다.
        chunk_words = words[start_index : start_index + chunk_size]

        # 더 이상 담을 내용이 없으면 반복을 끝낸다.
        if not chunk_words:
            # break 로 for 반복문을 중단해 빈 청크 생성 시도를 멈춘다.
            break

        # 단어들을 다시 문자열로 합쳐 하나의 chunk 를 만든다.
        chunk = " ".join(chunk_words)

        # 비어 있지 않은 chunk 만 결과에 추가한다.
        if chunk.strip():
            # chunks 리스트에 현재 chunk 문자열을 append 함수로 추가한다.
            chunks.append(chunk)

    # 생성된 chunk 리스트를 반환한다.
    return chunks


# 이 함수는 query 벡터와 여러 문서 벡터 사이의 내적 기반 점수를 계산해 상위 k 개를 반환한다.
def top_k_retrieval(query_vector: Sequence[float], document_vectors: Sequence[Sequence[float]], k: int = 3) -> List[Tuple[int, float]]:
    # 문서별 점수를 저장할 빈 리스트를 만든다.
    scored_documents = []

    # 각 문서 벡터를 순회하며 query 와의 단순 점수를 계산한다.
    for index, document_vector in enumerate(document_vectors):
        # 길이가 다르면 비교할 수 없으므로 예외를 발생시킨다.
        if len(query_vector) != len(document_vector):
            # ValueError 로 query_vector 와 document_vector 차원이 같아야 함을 알린다.
            raise ValueError("query_vector and document_vector must have the same length.")

        # 가장 단순한 예시로 내적 점수를 사용한다.
        score = sum(query_value * doc_value for query_value, doc_value in zip(query_vector, document_vector))

        # 문서 인덱스와 점수를 함께 저장한다.
        scored_documents.append((index, score))

    # 점수가 높은 순서대로 정렬한다.
    scored_documents.sort(key=lambda item: item[1], reverse=True)

    # 상위 k 개만 잘라서 반환한다.
    return scored_documents[:k]


# 이 아래 코드는 파일을 직접 실행했을 때 동작하는 간단한 예시다.
if __name__ == "__main__":
    # 예시 문서들을 만든다.
    sample_texts = [
        # 첫 번째 예시 문서는 RAG 개념을 짧게 설명한다.
        "RAG는 검색 증강 생성이다.",
        # 두 번째 예시 문서는 LLM 약어의 뜻을 설명한다.
        "LLM은 대규모 언어 모델이다.",
        # 세 번째 예시 문서는 임베딩의 역할을 설명한다.
        "임베딩은 의미를 숫자 벡터로 바꾼다.",
        # 이 줄은 sample_texts 예시 리스트 정의를 닫는다.
    ]

    # 단어 빈도를 출력한다.
    print("word_frequency:", word_frequency(sample_texts, top_k=5))

    # vocabulary 를 만든다.
    vocab = build_vocabulary(sample_texts)
    # build_vocabulary 함수가 만든 단어-인덱스 매핑을 print 함수로 출력한다.
    print("vocabulary:", vocab)

    # bag-of-words 벡터 예시를 출력한다.
    print("bow_vector:", text_to_bow_vector("rag와 임베딩을 공부한다", vocab))

    # chunking 예시를 출력한다.
    print("chunks:", chunk_text("이 문서는 RAG를 설명하기 위한 아주 짧은 예시 문장이다.", chunk_size=4, overlap=1))
