# 이 파일은 PPO 와 CQL 을 "숫자가 실제로 어떻게 움직이는지" 중심으로 읽어 보는 학습용 파일이다.
#
# PPO 는 Proximal Policy Optimization 의 약자다.
# 직역하면 "가까운 범위 안에서 정책을 최적화한다"는 뜻이고,
# 쉬운 말로는 정책을 한 번에 너무 크게 바꾸지 않으며 조금씩 고치는 방식이다.
#
# CQL 은 Conservative Q-Learning 의 약자다.
# 직역하면 "보수적인 Q 학습"이고,
# 쉬운 말로는 데이터에 없던 행동을 모델이 너무 좋게 착각하지 않도록 조심스럽게 배우는 방식이다.

# 로그, exp 같은 수학 함수를 쓰기 위해 math 모듈을 가져온다.
import math

# 타입 힌트를 위해 typing 도구를 가져온다.
# Sequence 는 list, tuple 처럼 순서가 있는 데이터 묶음을 뜻한다.
from typing import Dict, List, Sequence, Tuple


# 이 함수는 reward 목록을 받아 discounted return 을 계산한다.
# return 은 "지금 시점부터 앞으로 받을 보상을 한데 합친 값"이라고 생각하면 된다.
# gamma 는 미래 보상을 현재보다 얼마나 덜 중요하게 볼지 정하는 할인율이다.
def discounted_returns(rewards: Sequence[float], gamma: float = 0.99) -> List[float]:
    # reward 가 비어 있으면 계산할 흐름 자체가 없으므로 예외를 발생시킨다.
    if len(rewards) == 0:
        raise ValueError("rewards must not be empty.")

    # gamma 는 보통 0~1 사이를 사용한다.
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1.")

    # 결과를 같은 길이의 리스트로 미리 만들어 둔다.
    returns = [0.0 for _ in rewards]

    # 뒤에서부터 누적할 현재 return 값을 0으로 시작한다.
    running_return = 0.0

    # 맨 뒤 reward 부터 앞으로 거꾸로 오면서 누적 return 을 계산한다.
    for index in range(len(rewards) - 1, -1, -1):
        # 현재 보상 + 할인된 미래 보상을 더하면 현재 시점의 return 이 된다.
        running_return = rewards[index] + gamma * running_return
        returns[index] = running_return

    # 각 시점의 return 리스트를 반환한다.
    return returns


# 이 함수는 return 과 value estimate 를 이용해 advantage 를 계산한다.
# advantage 는 "이번 행동이 평소 기대치보다 얼마나 더 좋았는지"를 나타내는 값이다.
def advantages_from_baseline(
    returns: Sequence[float],
    value_estimates: Sequence[float],
) -> List[float]:
    # 두 입력의 길이가 다르면 같은 시점을 비교할 수 없으므로 예외를 발생시킨다.
    if len(returns) != len(value_estimates):
        raise ValueError("returns and value_estimates must have the same length.")

    # 비어 있는 데이터는 학습용 계산으로 보기 어렵기 때문에 예외를 발생시킨다.
    if len(returns) == 0:
        raise ValueError("returns must not be empty.")

    # return 에서 baseline 역할의 value estimate 를 빼면 advantage 가 된다.
    return [current_return - value for current_return, value in zip(returns, value_estimates)]


# 이 함수는 PPO 의 핵심인 clipped objective 를 한 step 기준으로 계산한다.
# 여기서 clip 은 clipping 의 줄임말처럼 쓰이며, 값을 일정 범위 밖으로 못 나가게 "잘라 두는 것"이다.
# objective 는 모델이 키우거나 줄이려고 하는 목표식이고,
# clipped objective 는 "너무 큰 정책 변화는 잘라 낸 목표식" 정도로 이해하면 된다.
# policy 는 "이 상태에서 어떤 행동을 얼마나 자주 할지 정하는 규칙"이라고 보면 된다.
# PPO 는 새 정책이 옛 정책에서 너무 멀리 튀지 않게 ratio 를 잘라서 학습을 안정화한다.
# 이름의 Proximal 은 "너무 멀리 가지 않고 가까운 범위 안에 머문다"는 느낌이다.
def ppo_clipped_objective(
    old_action_probability: float,
    new_action_probability: float,
    advantage: float,
    clip_epsilon: float = 0.2,
) -> Dict[str, float]:
    # 확률은 0보다 크고 1 이하여야 비율 계산이 안전하다.
    if not 0 < old_action_probability <= 1:
        raise ValueError("old_action_probability must be between 0 and 1.")

    if not 0 < new_action_probability <= 1:
        raise ValueError("new_action_probability must be between 0 and 1.")

    # clip_epsilon 의 epsilon 은 허용 오차나 작은 범위를 나타낼 때 자주 쓰는 문자다.
    # 여기서는 ratio 를 얼마나 넓게 허용할지 정하는 clip 폭이라고 보면 된다.
    # clip 범위는 음수가 될 수 없다.
    if clip_epsilon < 0:
        raise ValueError("clip_epsilon must be at least 0.")

    # ratio 는 "새 정책이 이 행동을 예전보다 얼마나 더/덜 밀어 주는가"를 보는 값이다.
    ratio = new_action_probability / old_action_probability

    # clip 을 하지 않으면 정책이 한 번에 너무 크게 바뀔 수 있다.
    unclipped_objective = ratio * advantage

    # PPO 는 ratio 를 1 - epsilon ~ 1 + epsilon 범위로 잘라 과격한 업데이트를 막는다.
    clipped_ratio = min(max(ratio, 1 - clip_epsilon), 1 + clip_epsilon)
    clipped_objective = clipped_ratio * advantage

    # surrogate objective 의 surrogate 는 "대리"라는 뜻이다.
    # 즉 진짜 최종 목표를 바로 쓰기보다, 안정적으로 학습하기 위한 대리 목표식을 쓴다는 뜻이다.
    # 둘 중 더 보수적인 값을 쓰는 것이 PPO 의 핵심 아이디어다.
    surrogate_objective = min(unclipped_objective, clipped_objective)

    # 중간 계산까지 같이 보여 주면 숫자 흐름을 이해하기 쉽다.
    return {
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "advantage": advantage,
        "unclipped_objective": unclipped_objective,
        "clipped_objective": clipped_objective,
        "surrogate_objective": surrogate_objective,
    }


# 이 함수는 여러 step 의 PPO 계산을 한 번에 묶어 보여 주는 학습용 헬퍼다.
# PPO 라는 이름이 길고 어렵게 보여도, 핵심은 "정책을 너무 급격히 바꾸지 않게 제한한다"는 데 있다.
def summarize_ppo_batch(
    old_action_probabilities: Sequence[float],
    new_action_probabilities: Sequence[float],
    advantages: Sequence[float],
    clip_epsilon: float = 0.2,
) -> Tuple[List[Dict[str, float]], float]:
    # 세 리스트 길이가 같아야 같은 step 끼리 대응해서 계산할 수 있다.
    if not (
        len(old_action_probabilities)
        == len(new_action_probabilities)
        == len(advantages)
    ):
        raise ValueError("All PPO input sequences must have the same length.")

    # 빈 배치는 평균 objective 를 계산할 수 없으므로 예외를 발생시킨다.
    if len(old_action_probabilities) == 0:
        raise ValueError("PPO input sequences must not be empty.")

    # step 별 계산 결과를 담을 리스트를 준비한다.
    step_summaries = []

    # 각 step 의 old/new 확률과 advantage 로 PPO objective 를 계산한다.
    for old_probability, new_probability, advantage in zip(
        old_action_probabilities,
        new_action_probabilities,
        advantages,
    ):
        step_summaries.append(
            ppo_clipped_objective(
                old_action_probability=old_probability,
                new_action_probability=new_probability,
                advantage=advantage,
                clip_epsilon=clip_epsilon,
            )
        )

    # 실제 PPO 는 여기에 value loss, entropy bonus 등을 더하기도 하지만,
    # 이 학습용 예시는 clip 된 policy objective 만 보여 주는 데 집중한다.
    mean_objective = sum(item["surrogate_objective"] for item in step_summaries) / len(step_summaries)

    # step 별 상세 결과와 평균 objective 를 함께 반환한다.
    return step_summaries, mean_objective


# 이 함수는 logsumexp 를 안정적으로 계산한다.
# CQL 에서는 여러 행동의 Q값을 한꺼번에 부드럽게 묶어 볼 때 이 계산이 자주 등장한다.
def stable_logsumexp(values: Sequence[float]) -> float:
    # 값이 없으면 합칠 대상이 없으므로 예외를 발생시킨다.
    if len(values) == 0:
        raise ValueError("values must not be empty.")

    # 가장 큰 값을 먼저 빼 두면 exp 계산이 너무 커지는 것을 줄일 수 있다.
    max_value = max(values)
    shifted_sum = sum(math.exp(value - max_value) for value in values)

    # 다시 max_value 를 더해 원래 스케일로 돌린다.
    return max_value + math.log(shifted_sum)


# 이 함수는 CQL 에서 쓰는 conservative penalty 를 계산한다.
# conservative 는 "보수적인, 조심스러운"이라는 뜻이다.
# penalty 는 잘못된 방향으로 가는 것을 막기 위해 더하는 벌점이라고 보면 된다.
# Q값의 Q 는 quality 에서 왔다고 이해하면 쉽고,
# "이 상태에서 이 행동을 하면 앞으로 얼마나 좋을까"를 점수로 적어 둔 값이다.
# Q-Learning 은 이런 Q값을 학습하는 강화학습 계열 방법이다.
# CQL 은 데이터셋에 있던 행동보다, 데이터셋에 없던 행동을 너무 좋게 보는 일을 경계한다.
def cql_conservative_penalty(q_values: Sequence[float], dataset_action_index: int) -> float:
    # 행동별 Q값이 비어 있으면 비교할 수 없으므로 예외를 발생시킨다.
    if len(q_values) == 0:
        raise ValueError("q_values must not be empty.")

    # 데이터셋에서 실제로 관측된 행동 인덱스가 범위를 벗어나면 잘못된 입력이다.
    if not 0 <= dataset_action_index < len(q_values):
        raise ValueError("dataset_action_index is out of range.")

    # 데이터셋에 실제로 있던 행동의 Q값을 꺼낸다.
    dataset_action_q = q_values[dataset_action_index]

    # 여러 행동 전체를 부드럽게 모은 값에서 데이터셋 행동 Q값을 빼면
    # 데이터에 없는 행동들을 모델이 얼마나 높게 보는지에 대한 벌점이 된다.
    return stable_logsumexp(q_values) - dataset_action_q


# 이 함수는 CQL 예시에서 사용할 1-step TD target 을 계산한다.
# TD 는 Temporal Difference 의 약자다.
# 직역하면 "시간 차이"이고, 지금 예측과 한 step 뒤 예측을 비교해 배우는 방식이라고 이해하면 된다.
# target 은 모델이 맞추려고 하는 목표값이다.
# done 은 이 step 에서 에피소드가 끝났는지를 뜻한다.
def cql_td_target(
    reward: float,
    next_q_values: Sequence[float],
    gamma: float = 0.99,
    done: bool = False,
) -> float:
    # gamma 는 보통 0~1 사이를 사용한다.
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1.")

    # 종료된 상태라면 미래 보상은 더 이어지지 않으므로 현재 reward 만 target 이 된다.
    if done:
        return reward

    # 종료되지 않았는데 다음 상태 Q값이 비어 있으면 target 을 만들 수 없다.
    if len(next_q_values) == 0:
        raise ValueError("next_q_values must not be empty when done is False.")

    # 가장 좋아 보이는 다음 행동의 Q값을 써서 1-step target 을 만든다.
    return reward + gamma * max(next_q_values)


# 이 함수는 CQL 의 한 step 손실을 학습용으로 단순화해 계산한다.
# 실제 구현은 더 복잡할 수 있지만, 여기서는 "TD 오차 + 보수적 벌점" 흐름을 보는 데 집중한다.
# Bellman error 는 강화학습에서 자주 보이는 오차 이름으로,
# 현재 Q값이 Bellman 식이 제시하는 목표값과 얼마나 차이 나는지 보는 값이다.
def cql_step_loss(
    q_values: Sequence[float],
    dataset_action_index: int,
    reward: float,
    next_q_values: Sequence[float],
    gamma: float = 0.99,
    alpha: float = 1.0,
    done: bool = False,
) -> Dict[str, float]:
    # alpha 는 conservative penalty 를 얼마나 강하게 줄지 정하는 가중치다.
    # 값이 클수록 CQL 이 더 보수적으로 학습된다고 볼 수 있다.
    if alpha < 0:
        raise ValueError("alpha must be at least 0.")

    # 데이터셋에서 실제로 관측된 행동의 현재 Q값을 가져온다.
    if not 0 <= dataset_action_index < len(q_values):
        raise ValueError("dataset_action_index is out of range.")

    dataset_action_q = q_values[dataset_action_index]

    # 다음 상태 정보로 1-step TD target 을 만든다.
    td_target = cql_td_target(
        reward=reward,
        next_q_values=next_q_values,
        gamma=gamma,
        done=done,
    )

    # Bellman error 는 현재 예측 Q값이 target 과 얼마나 차이 나는지 보는 항이다.
    bellman_error = (dataset_action_q - td_target) ** 2

    # conservative penalty 는 데이터에 없던 행동을 과대평가하지 않게 만드는 벌점이다.
    conservative_penalty = cql_conservative_penalty(
        q_values=q_values,
        dataset_action_index=dataset_action_index,
    )

    # CQL 은 보통 TD 학습 손실에 이 보수적 벌점을 더해 학습한다.
    # 즉 "정답 쪽으로 맞추기"와 "본 적 없는 행동을 너무 높게 보지 않기"를 함께 챙긴다.
    total_loss = bellman_error + alpha * conservative_penalty

    # 중간 계산을 함께 반환하면 어떤 항이 loss 를 키웠는지 읽기 좋다.
    return {
        "dataset_action_q": dataset_action_q,
        "td_target": td_target,
        "bellman_error": bellman_error,
        "conservative_penalty": conservative_penalty,
        "alpha": alpha,
        "total_loss": total_loss,
    }


# 이 함수는 여러 오프라인 샘플의 CQL 손실을 한 번에 묶어 보는 학습용 헬퍼다.
def summarize_cql_batch(
    q_value_rows: Sequence[Sequence[float]],
    dataset_action_indices: Sequence[int],
    rewards: Sequence[float],
    next_q_value_rows: Sequence[Sequence[float]],
    dones: Sequence[bool],
    gamma: float = 0.99,
    alpha: float = 1.0,
) -> Tuple[List[Dict[str, float]], float]:
    # 각 리스트 길이가 다르면 같은 샘플끼리 짝지어 계산할 수 없다.
    if not (
        len(q_value_rows)
        == len(dataset_action_indices)
        == len(rewards)
        == len(next_q_value_rows)
        == len(dones)
    ):
        raise ValueError("All CQL input sequences must have the same length.")

    # 샘플이 하나도 없으면 평균 loss 를 계산할 수 없다.
    if len(q_value_rows) == 0:
        raise ValueError("CQL input sequences must not be empty.")

    # 샘플별 상세 결과를 담을 리스트를 준비한다.
    step_summaries = []

    # 오프라인 배치의 각 샘플에 대해 CQL 손실을 계산한다.
    for q_values, action_index, reward, next_q_values, done in zip(
        q_value_rows,
        dataset_action_indices,
        rewards,
        next_q_value_rows,
        dones,
    ):
        step_summaries.append(
            cql_step_loss(
                q_values=q_values,
                dataset_action_index=action_index,
                reward=reward,
                next_q_values=next_q_values,
                gamma=gamma,
                alpha=alpha,
                done=done,
            )
        )

    # 샘플별 total loss 평균을 구해 배치 전체 흐름을 본다.
    mean_total_loss = sum(item["total_loss"] for item in step_summaries) / len(step_summaries)

    # 상세 결과와 평균 loss 를 함께 반환한다.
    return step_summaries, mean_total_loss


# 이 아래 코드는 파일을 직접 실행했을 때만 동작하는 간단한 PPO / CQL 예시다.
if __name__ == "__main__":
    # PPO 예시에서는 한 에피소드의 reward 흐름을 먼저 만든다.
    ppo_rewards = [1.0, 0.0, 2.0]

    # critic 은 actor-critic 구조에서 "가치를 평가하는 쪽"을 뜻한다.
    # actor 가 어떤 행동을 할지 고른다면, critic 은 그 상태나 행동이 얼마나 괜찮은지 점수를 매긴다.
    # 그래서 critic 이 각 시점에서 대충 예상했던 가치라고 생각하면 value estimate 를 이해하기 쉽다.
    ppo_value_estimates = [1.5, 2.0, 0.8]

    # return 과 advantage 를 차례대로 계산한다.
    ppo_returns = discounted_returns(ppo_rewards, gamma=0.9)
    ppo_advantages = advantages_from_baseline(ppo_returns, ppo_value_estimates)

    # old probability 는 예전 정책이 고른 행동의 확률이고,
    # new probability 는 업데이트 후 정책이 같은 행동에 주는 확률이다.
    ppo_old_action_probabilities = [0.40, 0.50, 0.25]
    ppo_new_action_probabilities = [0.52, 0.30, 0.34]

    # PPO clip 예시를 배치 단위로 요약한다.
    # 다시 말해 clipping 으로 정책 변화 폭을 제한했을 때 objective 가 어떻게 바뀌는지 본다.
    ppo_step_summaries, ppo_mean_objective = summarize_ppo_batch(
        old_action_probabilities=ppo_old_action_probabilities,
        new_action_probabilities=ppo_new_action_probabilities,
        advantages=ppo_advantages,
        clip_epsilon=0.2,
    )

    # 부동소수점은 컴퓨터가 소수(실수)를 저장하는 대표 방식이다.
    # "부동"은 소수점 위치가 고정되지 않고 움직일 수 있다는 뜻이고,
    # 그래서 아주 크거나 작은 수도 비슷한 방식으로 표현할 수 있다.
    # 다만 10진수 소수(예: 0.1)를 2진수로는 딱 떨어지게 저장하지 못하는 경우가 많아서
    # 0.30000000000000004 같은 미세한 오차가 보일 수 있다.
    # 여기서는 학습용 출력이 너무 지저분해지지 않도록 round 로 자릿수를 정리한다.
    rounded_ppo_step_summaries = [
        {key: round(value, 4) for key, value in summary.items()}
        for summary in ppo_step_summaries
    ]

    # PPO 예시 결과를 출력한다.
    print("ppo_returns:", [round(value, 4) for value in ppo_returns])
    print("ppo_advantages:", [round(value, 4) for value in ppo_advantages])
    print("ppo_step_summaries:", rounded_ppo_step_summaries)
    print("ppo_mean_objective:", round(ppo_mean_objective, 4))

    # CQL 예시에서는 이미 모여 있는 오프라인 로그 데이터 배치를 가정한다.
    # 오프라인 RL 은 새로 환경을 탐험하지 않고, 이미 저장된 기록만으로 배우는 설정이다.
    # 각 q_values 는 한 상태에서 가능한 행동들의 현재 Q값 예측이라고 보면 된다.
    cql_q_value_rows = [
        [1.2, 2.3, 1.6],
        [2.4, 1.1, 3.2],
    ]

    # dataset_action_indices 는 실제 데이터 로그에서 관측된 행동의 번호다.
    cql_dataset_action_indices = [1, 0]

    # rewards 는 실제로 받은 보상이고, dones 는 여기서 에피소드가 끝났는지 여부다.
    cql_rewards = [1.0, -0.3]
    cql_dones = [False, True]

    # next_q_value_rows 는 다음 상태에서의 Q값 예측이다.
    cql_next_q_value_rows = [
        [1.4, 2.1, 1.7],
        [0.4, 0.6, 0.2],
    ]

    # CQL 예시 손실을 계산한다.
    cql_step_summaries, cql_mean_total_loss = summarize_cql_batch(
        q_value_rows=cql_q_value_rows,
        dataset_action_indices=cql_dataset_action_indices,
        rewards=cql_rewards,
        next_q_value_rows=cql_next_q_value_rows,
        dones=cql_dones,
        gamma=0.9,
        alpha=0.5,
    )

    # CQL 결과도 같은 이유로 소수점 자릿수를 맞춰 읽기 쉽게 만든다.
    rounded_cql_step_summaries = [
        {key: round(value, 4) for key, value in summary.items()}
        for summary in cql_step_summaries
    ]

    # CQL 예시 결과를 출력한다.
    print("cql_step_summaries:", rounded_cql_step_summaries)
    print("cql_mean_total_loss:", round(cql_mean_total_loss, 4))
