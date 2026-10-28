# %%
# Converted from: 2.5 Matplotlib으로 시각화하기.ipynb
# Original .ipynb is preserved alongside this file.

# %%
# Jupyter-only command: !pip install matplotlib
# Jupyter-only command: !pip install numpy

# %%
import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(0, 10, 100)
y1 = x1 ** 2
y2 = 10 * x1 + 2

x2 = 10 + 10 * np.random.randn(1000)

# %%
# 셀 3: 두 개의 선 그래프 그리기

# plot(): x축 값과 y축 값을 이용해 선 그래프를 그린다.
# "r--"는 빨간색(red) 점선을 의미한다.
# label은 계산이나 선 모양을 바꾸지 않고 해당 선의 이름표를 지정한다.
plt.plot(x1, y1, "r--", label="y1")

# "b-"는 파란색(blue) 실선을 의미하고, 이 선의 이름은 y2이다.
plt.plot(x1, y2, "b-", label="y2")

# title(): 그래프의 제목을 지정한다.
plt.title("line plot")

# xlabel(), ylabel(): x축과 y축의 이름을 지정한다.
plt.xlabel("x")
plt.ylabel("y")

# legend(): 각 plot()에서 label로 지정한 선의 이름을 범례로 표시한다.
# label만 지정하고 legend()를 호출하지 않으면 이름표가 화면에 보이지 않는다.
# loc="upper left"는 범례를 그래프의 왼쪽 위에 배치한다.
plt.legend(loc="upper left")

# show(): 완성된 그래프를 화면에 출력한다.
plt.show()

# %%
# 셀 4: 그래프 영역을 둘로 나누기

# subplot(행 개수, 열 개수, 위치 번호):
# 전체 그래프 영역을 1행 2열로 나누고 첫 번째 영역을 선택한다.
plt.subplot(1, 2, 1)

# 첫 번째 영역에 y1을 초록색 점선으로 그린다.
plt.plot(x1, y1, "g--", label="y1")


# 1행 2열로 나눈 그래프에서 두 번째 영역을 선택한다.
plt.subplot(1, 2, 2)

# 두 번째 영역에 y2를 실선으로 그린다.
# 색상을 지정하지 않았으므로 Matplotlib의 기본 색상을 사용한다.
plt.plot(x1, y2, "-", label="y2")


# label은 지정했지만 legend()를 호출하지 않았으므로
# 현재 코드에서는 y1과 y2 이름이 범례로 표시되지 않는다.

plt.show()

# %%
# 셀 5: 히스토그램 그리기

# hist(): 숫자 데이터가 각 구간에 얼마나 분포하는지 히스토그램으로 나타낸다.
# x2는 히스토그램으로 표현할 난수 데이터이다.
# 50은 데이터 범위를 50개의 구간으로 나눈다는 의미이다.
# density=True는 단순 개수가 아니라 전체 면적이 1이 되도록 정규화한다.
# facecolor="orange"는 막대 내부 색상을 주황색으로 지정한다.
# alpha=0.75는 막대의 불투명도를 75%로 지정한다.
plt.hist(
    x2,
    50,
    density=True,
    facecolor="orange",
    alpha=0.75,
)

plt.show()

# %%
# 셀 6: 예제 이미지 출력

# skimage의 data 모듈에는 실습에 사용할 수 있는 예제 이미지가 포함되어 있다.
from skimage import data


# xticks([]): x축의 눈금과 눈금 값을 표시하지 않는다.
plt.xticks([])

# yticks([]): y축의 눈금과 눈금 값을 표시하지 않는다.
plt.yticks([])


# data.rocket(): skimage에서 제공하는 로켓 예제 이미지를 NumPy 배열로 반환한다.
# imshow(): 이미지 배열을 그래프 영역에 표시한다.
plt.imshow(data.rocket())

# 원본 노트북에서는 마지막 그래프가 자동 표시되지만,
# 일반 .py 파일에서도 확실히 창을 표시하도록 show()를 호출한다.
plt.show()

