# %%
# Converted from: 3.2 다층 신경망과 XOR 문제 해결하기.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# # 다층 신경망과 XOR 문제 해결하기

# %%
# 셀 1: 퍼셉트론으로 NAND 게이트 구현

def NAND_gate(x1, x2):
    # NAND 게이트는 두 입력이 모두 1인 경우에만 0을 출력해야 한다.
    # 입력값이 증가할수록 계산 결과가 작아지도록 음수 가중치를 사용한다.
    w1 = -0.5
    w2 = -0.5

    # 입력이 모두 0이거나 하나만 1일 때 결과가 양수가 되도록
    # bias를 0.6으로 설정한다.
    bias = 0.6

    # 입력값과 음수 가중치를 곱하고 bias를 더한다.
    result = x1 * w1 + x2 * w2 + bias

    # 계산 결과가 0보다 크면 1, 그렇지 않으면 0을 반환한다.
    if result > 0:
        return 1
    else:
        return 0


# NAND 게이트는 AND 게이트의 출력 결과를 반대로 만든다.
NAND_gate(0, 0), NAND_gate(0, 1), NAND_gate(1, 0), NAND_gate(1, 1)

# 출력 결과:
# (1, 1, 1, 0)
#
# NAND_gate(0, 0): 0×(-0.5) + 0×(-0.5) + 0.6 =  0.6 → 1
# NAND_gate(0, 1): 0×(-0.5) + 1×(-0.5) + 0.6 =  0.1 → 1
# NAND_gate(1, 0): 1×(-0.5) + 0×(-0.5) + 0.6 =  0.1 → 1
# NAND_gate(1, 1): 1×(-0.5) + 1×(-0.5) + 0.6 = -0.4 → 0

# %%
def OR_gate(x1, x2):
  # weights
  w1 = 0.5
  w2 = 0.5
  bias = -0.2

  # forward
  result = x1*w1 + x2*w2 + bias

  # activation
  if result > 0:
    return 1
  else:
    return 0

# %%
# 셀 3: XOR 신경망의 첫 번째 계층 구현

# step_function(): 계산값이 0보다 크면 1, 그렇지 않으면 0을 반환한다.
def step_function(x):
    return 1 if x > 0 else 0


def layer1(x1, x2):
    # 첫 번째 퍼셉트론은 NAND 게이트 역할을 한다.
    w00 = -0.5
    w01 = -0.5
    b0 = 0.6

    # 두 번째 퍼셉트론은 OR 게이트 역할을 한다.
    w10 = 0.6
    w11 = 0.6
    b1 = -0.5

    # 같은 입력 x1, x2를 NAND와 OR 퍼셉트론에 각각 전달한다.
    result0 = x1 * w00 + x2 * w01 + b0
    result1 = x1 * w10 + x2 * w11 + b1

    # 가중치 계산 결과를 0 또는 1로 변환한다.
    result0 = step_function(result0)
    result1 = step_function(result1)

    # NAND 결과와 OR 결과를 함께 반환한다.
    return result0, result1


# 입력별 반환 결과:
# layer1(0, 0) → (NAND=1, OR=0)
# layer1(0, 1) → (NAND=1, OR=1)
# layer1(1, 0) → (NAND=1, OR=1)
# layer1(1, 1) → (NAND=0, OR=1)

# %%
# 셀 4: 두 계층을 연결해 XOR 게이트 구현

def layer2(x1, x2):
    # 두 번째 계층은 첫 번째 계층에서 나온 NAND와 OR 결과를
    # AND 연산으로 결합하는 퍼셉트론이다.
    w0 = 0.5
    w1 = 0.5
    bias = -0.7

    # 두 입력이 모두 1일 때만 계산 결과가 0보다 커진다.
    result = x1 * w0 + x2 * w1 + bias

    return step_function(result)


# MLP는 여러 계층의 퍼셉트론을 연결한 다층 퍼셉트론을 의미한다.
def XOR_MLP(x1, x2):
    # 원래 입력 x1, x2를 첫 번째 계층에 전달한다.
    # layer1()은 NAND 결과와 OR 결과를 반환한다.
    # 여기서 x1과 x2는 원래 입력 대신 첫 번째 계층의 중간 결과로 다시 저장된다.
    x1, x2 = layer1(x1, x2)

    # 첫 번째 계층의 NAND와 OR 결과를 두 번째 계층에 전달한다.
    y = layer2(x1, x2)

    return y


XOR_MLP(0, 0), XOR_MLP(0, 1), XOR_MLP(1, 0), XOR_MLP(1, 1)

# 출력 결과:
# (0, 1, 1, 0)
#
# 원래 입력  → layer1(NAND, OR) → layer2(AND) → XOR 출력
# (0, 0)    → (1, 0)            → 0           → 0
# (0, 1)    → (1, 1)            → 1           → 1
# (1, 0)    → (1, 1)            → 1           → 1
# (1, 1)    → (0, 1)            → 0           → 0
#
# XOR 게이트는 두 입력이 서로 다를 때만 1을 반환한다.

