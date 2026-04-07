# 이 파일은 AI 코드에서 자주 등장하는 "기초 수학 + 데이터 처리" 함수를 모아 둔 학습용 파일이다.

# 이 모듈은 난수 고정, 랜덤 섞기, 간단한 샘플링에 쓰기 위해 random 모듈을 가져온다.
import random

# 이 모듈은 exp, sqrt 같은 수학 함수를 쓰기 위해 math 모듈을 가져온다.
import math

# 이 모듈은 함수의 입력과 출력 타입을 읽기 좋게 적기 위해 typing 도구를 가져온다.
from typing import Iterable, Iterator, List, Sequence, Tuple


# 이 함수는 실험을 다시 돌렸을 때 같은 결과가 나오도록 seed 를 고정한다.
def set_seed(seed: int) -> None:
    # random 모듈의 seed 를 고정하면 random.shuffle, random.random 같은 결과가 재현 가능해진다.
    random.seed(seed)


# 이 함수는 데이터를 train 과 validation 으로 나누는 아주 기본적인 버전이다.
def train_val_split(
    items: Sequence,
    val_ratio: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List, List]:
    # validation 비율이 0보다 작거나 1보다 크면 잘못된 입력이므로 예외를 발생시킨다.
    if not 0 <= val_ratio <= 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    # 원본 시퀀스를 바로 바꾸지 않기 위해 list 로 복사본을 만든다.
    copied_items = list(items)

    # 데이터 분할 전에 순서를 섞고 싶다면 seed 를 고정한 뒤 shuffle 한다.
    if shuffle:
        random.seed(seed)
        random.shuffle(copied_items)

    # validation 데이터 개수를 계산한다.
    val_size = int(len(copied_items) * val_ratio)

    # 뒤쪽 val_size 개를 validation 으로 쓰고, 나머지를 train 으로 사용한다.
    train_items = copied_items[:-val_size] if val_size > 0 else copied_items
    val_items = copied_items[-val_size:] if val_size > 0 else []

    # train 과 validation 리스트를 튜플로 반환한다.
    return train_items, val_items


# 이 함수는 데이터를 batch_size 단위로 잘라서 순서대로 내보낸다.
def batch_iterator(items: Sequence, batch_size: int) -> Iterator[List]:
    # batch 크기가 1보다 작으면 의미가 없으므로 예외를 발생시킨다.
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    # 0부터 전체 길이까지 batch_size 간격으로 이동하면서 배치를 만든다.
    for start_index in range(0, len(items), batch_size):
        # 현재 시작 위치부터 batch_size 만큼 잘라 하나의 배치를 만든다.
        batch = list(items[start_index : start_index + batch_size])

        # 만들어진 배치를 하나씩 바깥으로 내보낸다.
        yield batch


# 이 함수는 두 벡터의 내적을 계산한다.
def dot_product(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    # 두 벡터 길이가 다르면 내적을 정의하기 어렵기 때문에 예외를 발생시킨다.
    if len(vector_a) != len(vector_b):
        raise ValueError("Both vectors must have the same length.")

    # 각 위치의 값을 곱한 뒤 모두 더해서 내적을 만든다.
    return sum(a * b for a, b in zip(vector_a, vector_b))


# 이 함수는 벡터의 L2 norm, 즉 길이를 계산한다.
def l2_norm(vector: Sequence[float]) -> float:
    # 각 원소 제곱의 합을 구한 뒤 제곱근을 취하면 벡터 길이가 된다.
    return math.sqrt(sum(value * value for value in vector))


# 이 함수는 두 벡터가 얼마나 비슷한 방향을 보는지 cosine similarity 로 계산한다.
def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    # 벡터 길이가 다르면 비교할 수 없으므로 예외를 발생시킨다.
    if len(vector_a) != len(vector_b):
        raise ValueError("Both vectors must have the same length.")

    # 분자에는 두 벡터의 내적을 넣는다.
    numerator = dot_product(vector_a, vector_b)

    # 분모에는 각 벡터의 길이를 곱한 값을 넣는다.
    denominator = l2_norm(vector_a) * l2_norm(vector_b)

    # 길이가 0인 벡터가 들어오면 0으로 나누게 되므로 예외를 발생시킨다.
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    # 내적을 각 길이의 곱으로 나누면 cosine similarity 가 된다.
    return numerator / denominator


# 이 함수는 입력값을 0과 1 사이로 눌러 주는 sigmoid 함수다.
def sigmoid(x: float) -> float:
    # sigmoid 는 이진 분류 출력층에서 자주 등장하며, 값을 확률처럼 해석하기 좋게 만든다.
    return 1 / (1 + math.exp(-x))


# 이 함수는 음수는 0으로 만들고 양수는 그대로 두는 ReLU 함수다.
def relu(x: float) -> float:
    # ReLU 는 은닉층에서 매우 자주 쓰이는 활성화 함수다.
    return max(0.0, x)


# 이 함수는 여러 점수(logit)를 확률 분포처럼 바꾸는 softmax 함수다.
def softmax(logits: Sequence[float]) -> List[float]:
    # 빈 리스트가 들어오면 계산할 값이 없으므로 예외를 발생시킨다.
    if len(logits) == 0:
        raise ValueError("logits must not be empty.")

    # 수치적으로 더 안정적인 계산을 위해 가장 큰 값을 먼저 뺀다.
    max_logit = max(logits)

    # 각 logit 에 exp 를 적용해서 양수 값으로 바꾼다.
    exp_values = [math.exp(logit - max_logit) for logit in logits]

    # exp 값의 총합을 구한다.
    total = sum(exp_values)

    # 각 exp 값을 총합으로 나누면 전체 합이 1인 확률 분포가 된다.
    return [value / total for value in exp_values]


# 이 함수는 값을 0과 1 사이 구간으로 min-max 정규화한다.
def min_max_normalize(values: Sequence[float]) -> List[float]:
    # 빈 리스트는 정규화할 수 없으므로 예외를 발생시킨다.
    if len(values) == 0:
        raise ValueError("values must not be empty.")

    # 정규화에 필요한 최소값과 최대값을 구한다.
    min_value = min(values)
    max_value = max(values)

    # 모든 값이 같으면 분모가 0이 되므로, 이 경우 0.0 리스트를 반환한다.
    if min_value == max_value:
        return [0.0 for _ in values]

    # 각 값을 (x - min) / (max - min) 공식으로 정규화한다.
    return [(value - min_value) / (max_value - min_value) for value in values]


# 이 아래 코드는 이 파일을 직접 실행했을 때만 동작하는 간단한 사용 예시다.
if __name__ == "__main__":
    # 재현 가능한 셔플 결과를 보기 위해 seed 를 고정한다.
    set_seed(123)

    # 예시 데이터 리스트를 만든다.
    sample_data = [1, 2, 3, 4, 5, 6]

    # train/validation 분할 예시를 실행한다.
    train_data, val_data = train_val_split(sample_data, val_ratio=0.33, shuffle=True, seed=123)

    # 분할된 결과를 출력한다.
    print("train_data:", train_data)
    print("val_data:", val_data)

    # batch 단위로 데이터를 묶는 예시를 출력한다.
    print("batches:", list(batch_iterator(sample_data, batch_size=2)))

    # softmax 결과를 출력해 확률 분포처럼 바뀌는 모습을 확인한다.
    print("softmax:", softmax([2.0, 1.0, 0.1]))

    # cosine similarity 결과를 출력해 두 벡터의 방향 유사도를 본다.
    print("cosine_similarity:", cosine_similarity([1, 2, 3], [1, 2, 3]))
