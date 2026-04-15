# 이 파일은 학습 루프와 평가 단계에서 자주 보이는 함수를 모아 둔 학습용 파일이다.

# 로그 계산을 위해 math 모듈을 가져온다.
import math

# 타입 힌트를 위해 typing 도구를 가져온다.
# 여기서 Sequence 는 list, tuple처럼 순서가 있는 데이터 묶음을 뜻한다.
from typing import Dict, List, Sequence, Tuple


# 이 함수는 이진 분류에서 sigmoid 출력 확률에 대한 binary cross entropy 를 계산한다.
# 쉽게 말해 "모델이 낸 확률이 정답과 얼마나 어긋났는지"를 숫자로 바꾸는 함수다.
def binary_cross_entropy(probability: float, target: int, epsilon: float = 1e-12) -> float:
    # probability 는 보통 sigmoid 를 통과한 뒤의 값이라서 0~1 사이에 있다.
    # target 은 정답이 음성인지 양성인지만 나타내므로 0 또는 1 만 사용한다.

    # target 이 0 또는 1 이 아니면 이진 분류 타깃으로 보기 어렵기 때문에 예외를 발생시킨다.
    if target not in (0, 1):
        # ValueError 는 binary cross entropy 함수의 target 규칙 위반을 알려 준다.
        raise ValueError("target must be 0 or 1.")

    # log(0)을 피하기 위해 probability 를 아주 작은 범위 안으로 잘라 준다.
    clipped_probability = min(max(probability, epsilon), 1 - epsilon)

    # 정답인데 낮은 확률을 줬거나, 오답인데 높은 확률을 주면 loss 가 크게 나온다.
    # 즉 "확신을 가지고 틀릴수록" 더 크게 벌점을 준다고 이해하면 된다.
    loss = -(
        # 이 줄은 math.log 함수로 양성 클래스 쪽 로그 우도 항을 계산한다.
        target * math.log(clipped_probability)
        # 이 줄은 음성 클래스 쪽 로그 우도 항을 더해 BCE 수식을 완성한다.
        + (1 - target) * math.log(1 - clipped_probability)
        # 이 줄은 괄호로 묶은 binary cross entropy 식을 닫는다.
    )

    # 계산된 손실 값을 반환한다.
    return loss


# 이 함수는 softmax 확률 벡터와 정답 인덱스를 받아 cross entropy 를 계산한다.
# 여러 선택지 중 정답 칸에 모델이 얼마나 높은 확률을 줬는지 보고 손실을 계산한다.
def multiclass_cross_entropy(probabilities: Sequence[float], target_index: int, epsilon: float = 1e-12) -> float:
    # 확률 리스트가 비어 있으면 계산할 수 없으므로 예외를 발생시킨다.
    if len(probabilities) == 0:
        # multiclass cross entropy 는 최소 한 개 이상의 클래스 확률이 필요하다.
        raise ValueError("probabilities must not be empty.")

    # 정답 인덱스가 범위를 벗어나면 잘못된 입력이므로 예외를 발생시킨다.
    if not 0 <= target_index < len(probabilities):
        # ValueError 로 target_index 가 probabilities 길이 안에 있어야 함을 알린다.
        raise ValueError("target_index is out of range.")

    # 정답 클래스의 확률만 꺼낸다.
    target_probability = probabilities[target_index]

    # log(0)을 피하기 위해 확률을 살짝 잘라 준다.
    clipped_probability = min(max(target_probability, epsilon), 1 - epsilon)

    # 정답 클래스 확률의 음의 로그를 반환한다.
    return -math.log(clipped_probability)


# 이 함수는 분류 정확도를 계산한다.
def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    # 정답과 예측 길이가 다르면 한 쌍씩 비교할 수 없으므로 예외를 발생시킨다.
    if len(y_true) != len(y_pred):
        # accuracy 계산은 위치별 비교이므로 두 시퀀스 길이가 같아야 한다.
        raise ValueError("y_true and y_pred must have the same length.")

    # 데이터가 없으면 정확도를 정의하기 애매하므로 예외를 발생시킨다.
    if len(y_true) == 0:
        # 빈 입력에서는 분모가 0이 되므로 ValueError 로 미리 막는다.
        raise ValueError("y_true must not be empty.")

    # 같은 위치에서 정답과 예측이 맞은 개수를 센다.
    correct_count = sum(1 for true_value, pred_value in zip(y_true, y_pred) if true_value == pred_value)

    # 전체 개수로 나누면 정확도가 된다.
    return correct_count / len(y_true)


# 이 함수는 이진 분류용 confusion matrix 를 딕셔너리로 반환한다.
# confusion matrix 는 예측 결과를 "맞춘 양성, 틀린 양성, 맞춘 음성, 놓친 양성"으로 나눠 세는 표다.
def confusion_matrix_binary(y_true: Sequence[int], y_pred: Sequence[int], positive_label: int = 1) -> Dict[str, int]:
    # 길이가 다르면 비교할 수 없으므로 예외를 발생시킨다.
    if len(y_true) != len(y_pred):
        # confusion matrix 도 정답과 예측을 짝지어 세기 때문에 길이가 같아야 한다.
        raise ValueError("y_true and y_pred must have the same length.")

    # confusion matrix 의 네 칸을 0으로 시작한다.
    tp = fp = tn = fn = 0

    # 정답과 예측을 한 쌍씩 보며 어느 칸에 들어가는지 센다.
    for true_value, pred_value in zip(y_true, y_pred):
        # 양성으로 예측했고 실제도 양성이면 true positive 다.
        if pred_value == positive_label and true_value == positive_label:
            # tp 카운터를 1 늘려 맞춘 양성 개수를 누적한다.
            tp += 1
        # 양성으로 예측했지만 실제는 음성이면 false positive 다.
        elif pred_value == positive_label and true_value != positive_label:
            # fp 카운터를 1 늘려 틀린 양성 예측 개수를 센다.
            fp += 1
        # 음성으로 예측했고 실제도 음성이면 true negative 다.
        elif pred_value != positive_label and true_value != positive_label:
            # tn 카운터를 1 늘려 맞춘 음성 개수를 누적한다.
            tn += 1
        # 음성으로 예측했지만 실제는 양성이면 false negative 다.
        else:
            # fn 카운터를 1 늘려 놓친 양성 개수를 센다.
            fn += 1

    # 보기 쉬운 키 이름으로 결과를 반환한다.
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


# 이 함수는 precision, recall, F1 을 한 번에 계산한다.
# precision 은 "양성이라고 한 것의 정확도", recall 은 "실제 양성을 얼마나 놓치지 않았는지"를 본다.
# F1 은 둘 중 하나만 좋다고 끝내지 않고, 두 값의 균형을 같이 보려는 지표다.
def precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int], positive_label: int = 1) -> Dict[str, float]:
    # 먼저 confusion matrix 를 계산해 기본 재료를 만든다.
    matrix = confusion_matrix_binary(y_true, y_pred, positive_label=positive_label)

    # 분자를 꺼내 쓰기 쉽게 변수에 담는다.
    tp = matrix["tp"]
    # fp 값은 false positive 개수라 precision 계산에 쓰인다.
    fp = matrix["fp"]
    # fn 값은 false negative 개수라 recall 계산에 쓰인다.
    fn = matrix["fn"]

    # precision 은 양성이라고 한 것 중 실제 양성 비율이다.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # recall 은 실제 양성 중 찾아낸 비율이다.
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 은 precision 과 recall 의 조화평균이다.
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 세 지표를 딕셔너리 형태로 묶어 반환한다.
    return {"precision": precision, "recall": recall, "f1": f1}


# 이 함수는 학습 중 손실이 좋아졌는지, patience 를 얼마나 쌓았는지 판단하는 단순 early stopping 헬퍼다.
# patience 는 "조금 흔들려도 몇 번까지는 더 기다려 볼지"를 나타내는 여유 횟수다.
def update_early_stopping(
    # 이 매개변수는 지금까지 기록한 최고 성능 또는 최저 손실 값을 담는다.
    best_score: float,
    # 이 매개변수는 이번 평가에서 새로 측정된 점수다.
    current_score: float,
    # 이 매개변수는 개선 없이 지나간 횟수를 세는 patience 카운터다.
    patience_counter: int,
    # 이 매개변수는 점수가 클수록 좋은지 작을수록 좋은지 max/min 으로 정한다.
    mode: str = "max",
    # 이 매개변수는 개선으로 인정할 최소 변화량을 뜻한다.
    min_delta: float = 0.0,
    # 이 반환 타입 표시는 갱신된 best_score, patience 카운터, 개선 여부를 함께 돌려준다는 뜻이다.
) -> Tuple[float, int, bool]:
    # mode 는 max 또는 min 만 허용한다.
    if mode not in ("max", "min"):
        # ValueError 로 early stopping 모드가 두 가지 중 하나여야 함을 알린다.
        raise ValueError("mode must be 'max' or 'min'.")

    # 개선 여부를 먼저 False 로 시작한다.
    # early stopping 은 성능이 잠깐 출렁일 수 있다는 점을 고려해 한 번 나빠졌다고 바로 멈추지 않는다.
    improved = False

    # 성능이 클수록 좋은 상황이면 current_score 가 best_score 보다 충분히 커졌는지 본다.
    if mode == "max":
        # max 모드에서는 current_score 와 best_score 를 비교해 개선 여부를 계산한다.
        improved = current_score > best_score + min_delta
    # 손실처럼 작을수록 좋은 상황이면 current_score 가 best_score 보다 충분히 작아졌는지 본다.
    else:
        # min 모드에서는 current_score 가 더 작아졌는지를 기준으로 개선 여부를 본다.
        improved = current_score < best_score - min_delta

    # 개선됐다면 best_score 를 갱신하고 patience 카운터를 0으로 되돌린다.
    if improved:
        # 개선이 확인되면 새 best_score 와 초기화된 카운터, True 를 묶어 반환한다.
        return current_score, 0, True

    # 개선되지 않았다면 patience 카운터를 1 늘린다.
    return best_score, patience_counter + 1, False


# 이 아래 코드는 파일을 직접 실행했을 때 동작하는 작은 예시다.
if __name__ == "__main__":
    # 예시용 정답과 예측을 만든다.
    y_true_example = [1, 0, 1, 1, 0, 1]
    # 이 리스트는 모델이 낸 예측 라벨 예시다.
    y_pred_example = [1, 0, 0, 1, 0, 1]

    # 정확도를 계산해서 출력한다.
    print("accuracy:", accuracy_score(y_true_example, y_pred_example))

    # precision, recall, f1 을 계산해서 출력한다.
    print("metrics:", precision_recall_f1(y_true_example, y_pred_example))

    # confusion matrix 도 함께 출력한다.
    print("confusion_matrix:", confusion_matrix_binary(y_true_example, y_pred_example))

    # early stopping 업데이트 예시를 실행한다.
    print("early_stopping:", update_early_stopping(best_score=0.80, current_score=0.82, patience_counter=2))
