# %%
# Converted from: 3.1 퍼셉트론의 기본 개념.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# # 3.1 퍼셉트론의 기본 개념

# %% [markdown]
# #### AND 게이트

# %%
# 셀 1: 퍼셉트론으로 AND 게이트 구현

def AND_gate(x1, x2):
    # w1과 w2는 각 입력값의 중요도를 나타내는 가중치이다.
    w1 = 0.5
    w2 = 0.5

    # bias는 결과가 1이 되기 위한 기준을 조절한다.
    bias = -0.6

    # 각 입력값에 가중치를 곱한 뒤 bias를 더한다.
    result = x1 * w1 + x2 * w2 + bias

    # 계산 결과가 0보다 크면 1을 반환하고,
    # 0 이하이면 0을 반환하는 활성화 과정이다.
    if result > 0:
        return 1
    else:
        return 0


# AND 게이트는 두 입력이 모두 1일 때만 1을 반환한다.
AND_gate(0, 0), AND_gate(0, 1), AND_gate(1, 0), AND_gate(1, 1)

# 출력 결과:
# (0, 0, 0, 1)
#
# AND_gate(0, 0): 0×0.5 + 0×0.5 - 0.6 = -0.6 → 0
# AND_gate(0, 1): 0×0.5 + 1×0.5 - 0.6 = -0.1 → 0
# AND_gate(1, 0): 1×0.5 + 0×0.5 - 0.6 = -0.1 → 0
# AND_gate(1, 1): 1×0.5 + 1×0.5 - 0.6 =  0.4 → 1

# %% [markdown]
# #### OR 게이트

# %%
# 셀 2: 퍼셉트론으로 OR 게이트 구현

def OR_gate(x1, x2):
    # 두 입력 중 하나만 1이어도 결과가 기준을 넘도록
    # 각 입력의 가중치를 0.6으로 설정한다.
    w1 = 0.6
    w2 = 0.6

    # bias는 출력이 1로 바뀌는 기준을 조절한다.
    bias = -0.5

    # 입력값과 가중치를 곱하고 bias를 더한다.
    result = x1 * w1 + x2 * w2 + bias

    # 계산 결과가 0보다 크면 1, 그렇지 않으면 0을 반환한다.
    if result > 0:
        return 1
    else:
        return 0


# OR 게이트는 두 입력 중 하나 이상이 1이면 1을 반환한다.
OR_gate(0, 0), OR_gate(0, 1), OR_gate(1, 0), OR_gate(1, 1)

# 출력 결과:
# (0, 1, 1, 1)
#
# OR_gate(0, 0): 0×0.6 + 0×0.6 - 0.5 = -0.5 → 0
# OR_gate(0, 1): 0×0.6 + 1×0.6 - 0.5 =  0.1 → 1
# OR_gate(1, 0): 1×0.6 + 0×0.6 - 0.5 =  0.1 → 1
# OR_gate(1, 1): 1×0.6 + 1×0.6 - 0.5 =  0.7 → 1

# %% [markdown]
# #### NOT 게이트

# %%
# 셀 3: 퍼셉트론으로 NOT 게이트 구현

def NOT_gate(x1):
    # NOT 게이트는 입력과 반대되는 값을 출력해야 하므로
    # 입력이 커질수록 계산 결과가 작아지도록 음수 가중치를 사용한다.
    w1 = -0.6

    # 입력이 0일 때 결과가 양수가 되도록 bias를 0.5로 설정한다.
    bias = 0.5

    # 입력값에 가중치를 곱하고 bias를 더한다.
    result = x1 * w1 + bias

    # 계산 결과가 0보다 크면 1, 그렇지 않으면 0을 반환한다.
    if result > 0:
        return 1
    else:
        return 0


# NOT 게이트는 입력값을 반대로 바꿔 반환한다.
NOT_gate(0), NOT_gate(1)

# 출력 결과:
# (1, 0)
#
# NOT_gate(0): 0×(-0.6) + 0.5 =  0.5 → 1
# NOT_gate(1): 1×(-0.6) + 0.5 = -0.1 → 0

