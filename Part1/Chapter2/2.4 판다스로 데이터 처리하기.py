# %%
# Converted from: 2.4 판다스로 데이터 처리하기.ipynb
# Original .ipynb is preserved alongside this file.

# %%
# 셀 1: DataFrame 생성

import pandas as pd
import numpy as np
from IPython.display import display


my_2darray = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

# pd.DataFrame(): 배열이나 딕셔너리 등의 데이터를 행과 열로 구성된 표로 만든다.
# display(): 주피터 노트북에서 데이터를 표 형태로 출력한다.
display(pd.DataFrame(my_2darray))


my_dict = {
    "a": ["1", "3"],
    "b": ["1", "2"],
    "c": ["2", "4"],
}
display(pd.DataFrame(my_dict))


# data, index, columns를 직접 지정할 수 있다.
my_df = pd.DataFrame(
    data=[4, 5, 6, 7],
    index=range(0, 4),
    columns=["A"],
)
display(my_df)


# pd.Series(): 인덱스를 가진 1차원 데이터를 만든다.
my_series = pd.Series({
    "United Kingdom": "London",
    "India": "New Delhi",
    "United States": "Washington",
    "Belgium": "Brussels",
})
display(pd.DataFrame(my_series))

# %%
import pandas as pd
import numpy as np


# Series는 Pandas가 제공하는 1차원 자료형이다.
# 딕셔너리처럼 인덱스 이름으로 값을 찾을 수 있고,
# 리스트처럼 iloc을 사용해 위치 번호로도 값을 찾을 수 있다.
# 연산자와 Pandas·NumPy 함수를 전체 값에 한 번에 적용할 수 있다.
scores = pd.Series({
    "철수": 80,
    "영희": 90,
    "민수": 70,
})

# loc은 인덱스 이름으로 값을 선택한다.
print(scores.loc["영희"])

# iloc은 정수 위치로 값을 선택한다.
print(scores.iloc[1])

# Series의 모든 값에 10을 한 번에 더한다.
print(scores + 10)

# 조건을 만족하는 값만 선택한다.
print(scores[scores >= 80])

# Pandas의 통계 메서드를 전체 값에 적용한다.
print(scores.mean())

# NumPy 함수를 Series의 모든 값에 적용한다.
print(np.sqrt(scores))


# DataFrame은 Pandas가 제공하는 2차원 표 자료형이다.
# 각 열은 하나의 Series이며, 여러 Series가 모여 DataFrame을 구성한다.
students = pd.DataFrame({
    "이름": ["철수", "영희", "민수"],
    "점수": [80, 90, 70],
})

print(students)

# %%
# 셀 2: 행과 열 정보 확인

df = pd.DataFrame(
    np.array([
        [1, 2, 3],
        [4, 5, 6],
    ]),
    columns=["a", "b", "c"],
)

display(df)

# 출력 결과:
#    a  b  c
# 0  1  2  3
# 1  4  5  6


# shape은 DataFrame의 구조를 (행 개수, 열 개수) 형태로 반환한다.
print(df.shape)

# 출력 결과:
# (2, 3)


# index는 행의 인덱스 정보를 갖는다.
# len(df.index)는 행의 개수를 반환한다.
# list(df.index)는 행 인덱스를 파이썬 리스트로 변환한다.
print(len(df.index), list(df.index))

# 출력 결과:
# 2 [0, 1]


# columns는 열 이름 정보를 갖는다.
# len(df.columns)는 열의 개수를 반환한다.
# list(df.columns)는 열 이름을 파이썬 리스트로 변환한다.
print(len(df.columns), list(df.columns))

# 출력 결과:
# 3 ['a', 'b', 'c']

# %%
# 셀 3: 특정 열을 인덱스로 설정

df = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Gyuri", "Eunseop", "Mong"],
    "city": ["Seoul", "Incheon", "Suwon"],
    "item": ["jewelry", "Laptop", "snack"],
})

# set_index(): 지정한 열을 행 인덱스로 설정한다.
df = df.set_index("id")

display(df)


# %%
# 셀 4: loc으로 데이터 선택

# loc은 인덱스 이름과 열 이름을 이용해 원하는 데이터를 선택한다.
# 데이터를 선택해서 출력할 뿐 원본 df를 변경하지 않는다.


# 인덱스 이름이 1인 행 하나를 선택한다.
# 행을 단일 값으로 선택하면 결과는 Series로 반환된다.
display(df.loc[1])

# 출력 결과:
# name      Gyuri
# city      Seoul
# item    jewelry
# Name: 1, dtype: object


# 인덱스 이름이 1과 3인 행을 선택한다.
# 행 이름을 리스트로 전달하면 결과는 DataFrame으로 반환된다.
display(df.loc[[1, 3]])

# 출력 결과:
#      name   city     item
# id
# 1   Gyuri  Seoul  jewelry
# 3    Mong  Suwon    snack


# 콜론(:)은 모든 행을 의미한다.
# 모든 행에서 name 열만 선택한다.
# 열 이름을 리스트로 전달했으므로 결과는 DataFrame으로 반환된다.
display(df.loc[:, ["name"]])

# 출력 결과:
#        name
# id
# 1     Gyuri
# 2   Eunseop
# 3      Mong


# 인덱스 이름이 1과 3인 행에서 name과 item 열만 선택한다.
# loc의 기본 구조는 df.loc[선택할 행, 선택할 열]이다.
display(df.loc[[1, 3], ["name", "item"]])

# 출력 결과:
#      name     item
# id
# 1   Gyuri  jewelry
# 3    Mong    snack

# %%
# 셀 5: loc과 iloc의 차이

# loc은 현재 설정된 인덱스 이름과 열 이름을 기준으로 선택한다.
# 현재 df는 id 열을 인덱스로 설정했으므로 행 인덱스 이름은 1, 2, 3이다.
display(df.loc[1])

# 출력 결과:
# name      Gyuri
# city      Seoul
# item    jewelry
# Name: 1, dtype: object


# iloc은 인덱스 이름과 관계없이 0부터 시작하는 위치 번호로 선택한다.
# iloc[0]은 첫 번째 위치의 행이므로 현재 loc[1]과 같은 행을 선택한다.
display(df.iloc[0])

# 출력 결과:
# name      Gyuri
# city      Seoul
# item    jewelry
# Name: 1, dtype: object


# 같은 숫자 1을 사용해도 선택 기준이 다르다.
# loc[1]은 인덱스 이름이 1인 행을 선택한다.
display(df.loc[1])

# iloc[1]은 위치 번호가 1인 두 번째 행을 선택한다.
display(df.iloc[1])

# df.loc[1] 출력:
# name      Gyuri
# city      Seoul
# item    jewelry
# Name: 1, dtype: object

# df.iloc[1] 출력:
# name    Eunseop
# city    Incheon
# item     Laptop
# Name: 2, dtype: object


# loc은 행과 열의 이름을 사용한다.
display(df.loc[[1, 3], ["name", "item"]])

# iloc은 행과 열의 위치 번호를 사용한다.
# 행 위치 0과 2, 열 위치 0과 2를 선택한다.
display(df.iloc[[0, 2], [0, 2]])

# 두 코드의 출력 결과:
#      name     item
# id
# 1   Gyuri  jewelry
# 3    Mong    snack


# 범위를 선택할 때 loc은 마지막 인덱스 이름을 포함한다.
display(df.loc[1:2])

# 범위를 선택할 때 iloc은 마지막 위치 번호를 포함하지 않는다.
display(df.iloc[0:2])

# 두 코드 모두 현재 데이터에서는 첫 번째와 두 번째 행을 출력한다.
#        name     city     item
# id
# 1     Gyuri    Seoul  jewelry
# 2   Eunseop  Incheon   Laptop

# %%
# 셀 6: 열 이름으로 데이터 선택

# 열 하나를 지정하면 Series로 반환된다.
display(df["name"])

# 여러 열을 리스트로 지정하면 DataFrame으로 반환된다.
display(df[["name", "city"]])


# %%
# 셀 7: 행 인덱스를 설정하는 방법

df = pd.DataFrame({
    "A": [1, 4, 7],
    "B": [2, 5, 8],
    "C": [3, 6, 9],
})

display(df)

# 출력 결과:
#    A  B  C
# 0  1  2  3
# 1  4  5  6
# 2  7  8  9
#
# 현재 A, B, C는 모두 일반 데이터 열이다.
# 왼쪽의 0, 1, 2는 Pandas가 자동으로 만든 기본 행 인덱스이다.


# set_index("A")는 기존 A 열의 값 1, 4, 7을 행 인덱스로 이동한다.
# A 열이 실제 행으로 변하는 것이 아니라, A 열의 값이 각 행을 구분하는 이름표가 된다.
# 기본 설정에서는 A가 일반 데이터 열에서 제거되고 인덱스로 사용된다.
df = df.set_index("A")

display(df)

# 출력 결과:
#    B  C
# A
# 1  2  3
# 4  5  6
# 7  8  9
#
# 행 인덱스: 1, 4, 7
# 행 인덱스 이름: A
# 데이터 열: B, C


# DataFrame을 만들 때 index 매개변수를 사용하면
# 행 인덱스를 처음부터 직접 지정할 수 있다.
# index=[1, 4, 7]은 열의 인덱스가 아니라 각 행의 인덱스 이름이다.
df = pd.DataFrame(
    {
        "B": [2, 5, 8],
        "C": [3, 6, 9],
    },
    index=[1, 4, 7],
)

display(df)

# 출력 결과:
#    B  C
# 1  2  3
# 4  5  6
# 7  8  9
#
# 행 인덱스는 1, 4, 7이고 데이터 열은 B, C이다.
# 아직 행 인덱스 자체의 이름은 설정되지 않았다.


# index.name은 행 인덱스의 제목을 지정한다.
# A라는 데이터 열을 추가하는 것이 아니라,
# 왼쪽의 행 인덱스 1, 4, 7이 무엇을 나타내는지 이름을 붙이는 것이다.
df.index.name = "A"

display(df)

# 출력 결과:
#    B  C
# A
# 1  2  3
# 4  5  6
# 7  8  9
#
# 행 인덱스: 1, 4, 7
# 행 인덱스 이름: A
# 데이터 열: B, C

# %%
# 셀 8: 행 추가 및 수정

df = pd.DataFrame({
    "A": [1, 4, 7],
    "B": [2, 5, 8],
})
display(df)


# 존재하지 않는 인덱스를 loc으로 지정하면 새 행이 추가된다.
df.loc[3] = {
    "A": 10,
    "B": 11,
}
display(df)


# 현재 행 개수를 새로운 인덱스로 사용해 행을 추가한다.
df.loc[len(df.index)] = [50, 51]
display(df)


# 이미 존재하는 인덱스를 지정하면 해당 행의 값이 수정된다.
df.loc[0] = [100, 101]
display(df)

# %%
# 셀 9: 열 추가 및 수정

df = pd.DataFrame({
    "A": [1, 4, 7],
    "B": [2, 5, 8],
})
display(df)

# 현재 열 위치는 0번이 A, 1번이 B이다.
# iloc은 이미 존재하는 위치를 선택하거나 수정할 수 있지만
# 존재하지 않는 위치를 지정해 DataFrame의 크기를 늘릴 수는 없다.
# 따라서 아래 코드는 위치 2가 아직 존재하지 않아 오류가 발생한다.
# df.iloc[:, 2] = [3, 6, 9]  # IndexError: iloc cannot enlarge its target object


# 존재하지 않는 열 이름에 값을 대입하면 새 열이 추가된다.
df["C"] = [3, 6, 9]
display(df)


# loc을 이용해서도 모든 행에 새 열을 추가할 수 있다.
df.loc[:, "D"] = [4, 7, 10]
display(df)


# 기존 열에 값을 대입하면 해당 열의 값이 수정된다.
df.loc[:, "A"] = [11, 14, 17]
display(df)

# A 열은 이미 존재하므로 위치 번호를 사용하는 iloc으로도 수정할 수 있다.
# 다음 코드는 위의 loc 코드와 같은 A 열을 선택한다.
# df.iloc[:, 0] = [11, 14, 17]

# 정리:
# loc은 이름을 기준으로 선택하므로 새 행이나 새 열을 추가할 수 있다.
# iloc은 위치를 기준으로 선택하므로 기존 행이나 열만 선택·수정할 수 있다.

# %%
# 셀 10: 행과 열 삭제

df = pd.DataFrame({
    "A": [1, 4, 7],
    "B": [2, 5, 8],
})
display(df)


# drop(): 지정한 행이나 열을 삭제한다.
# 기본값 axis=0이므로 인덱스 이름이 1인 행을 삭제한다.
# inplace=False가 기본값이므로 삭제 결과가 적용된 새 DataFrame을 반환한다.
# 반환된 새 DataFrame을 다시 df에 저장하고 있으므로 이후 df가 변경된다.
df = df.drop(1)
display(df)


# inplace=False를 사용하면 기존 df는 직접 변경되지 않는다.
# 반환값을 새 변수에 저장하면 원본과 변경 결과를 모두 보관할 수 있다.
# new_df = df.drop("A", axis=1, inplace=False)
# display(df)      # 기존 A 열이 그대로 존재한다.
# display(new_df)  # A 열이 삭제된 새로운 DataFrame이다.

# 반환값을 저장하지 않으면 기존 df에는 아무 변화가 없다.
# df.drop("A", axis=1, inplace=False)

# 새 DataFrame을 같은 변수에 다시 저장하는 방식도 사용할 수 있다.
# df = df.drop("A", axis=1, inplace=False)


# axis=1은 행이 아니라 열을 대상으로 한다.
# inplace=True는 새로운 DataFrame을 반환하지 않고 원본 df를 직접 변경한다.
# inplace=True를 사용한 drop()의 반환값은 None이다.
df.drop("A", axis=1, inplace=True)
display(df)

# %%
# 셀 11: 중복 인덱스 제거

df = pd.DataFrame(
    data=np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [11, 12, 13],
        [21, 22, 23],
    ]),
    index=[1, 1, 2, 3, 4],
    columns=["A", "B", "C"],
)
display(df)

# 왼쪽의 1, 1, 2, 3, 4는 눈에는 보이지만 일반 데이터 열이 아니다.
# DataFrame이 별도로 관리하는 행 인덱스이며, 현재 인덱스 1이 중복되어 있다.
# 이 상태의 일반 데이터 열은 A, B, C뿐이다.


# reset_index(): 기존 행 인덱스를 index라는 일반 데이터 열로 이동한다.
# 동시에 0부터 시작하는 새로운 기본 행 인덱스를 만든다.
# inplace=True이므로 원본 df를 직접 변경한다.
df.reset_index(inplace=True)
display(df)

# reset_index() 실행 후 구조:
#   왼쪽의 0, 1, 2, 3, 4 → 새로 생성된 기본 행 인덱스
#   index 열의 1, 1, 2, 3, 4 → 이전의 행 인덱스 값


# drop_duplicates(): 중복된 행을 제거한다.
# subset에는 중복 판단에 사용할 일반 데이터 열 이름을 지정한다.
# reset_index() 전에는 index가 일반 열이 아니므로 subset="index"를 사용할 수 없다.
# reset_index()를 통해 index가 일반 열이 되었으므로 이제 중복 기준으로 사용할 수 있다.
# keep="last"는 중복된 index 값 중 마지막 행을 남긴다.
df = df.drop_duplicates(
    subset="index",
    keep="last",
)

# set_index("index")는 일반 열이 된 index 열을 다시 행 인덱스로 이동한다.
# 이 예제에서는 reset_index()로 꺼낸 인덱스를 원래 자리로 돌려놓는 역할을 한다.
# 하지만 중복 행 하나를 이미 삭제했으므로 삭제 전 상태를 완전히 복구하는 것은 아니다.
# set_index()는 반드시 복구용이 아니라 원하는 일반 열을 행 인덱스로 지정하는 함수이다.
df = df.set_index("index")
display(df)

# 최종 행 인덱스는 1, 2, 3, 4이며, 중복된 인덱스 1에서는 마지막 행이 유지된다.


# 행 인덱스 전용 duplicated()를 사용하면 reset_index() 없이도 처리할 수 있다.
# 다음 코드는 인덱스의 중복 여부를 직접 검사하고 마지막 항목만 유지한다.
# df = df[~df.index.duplicated(keep="last")]

# %%
# 셀 12: 결측값 제거와 인덱스 재설정

df = pd.DataFrame(
    data=np.array([
        [1, np.nan, 3],
        [1, 5, np.nan],
        [7, 8, 9],
    ]),
    columns=["A", "B", "C"],
)
display(df)

# 처음 행 인덱스는 0, 1, 2이다.
# np.nan은 값이 없음을 나타내는 결측값이다.


# dropna(): 하나 이상의 결측값이 포함된 행을 제거한다.
# 인덱스 0과 1의 행이 제거되고 기존 인덱스 2인 행만 남는다.
# 행을 삭제해도 남은 행의 기존 인덱스 2는 자동으로 0이 되지 않는다.
df = df.dropna()
display(df)

# dropna() 실행 후 결과:
#      A    B    C
# 2  7.0  8.0  9.0


# reset_index(): 기존 행 인덱스를 정리하고 0부터 시작하는 새 인덱스를 부여한다.
# drop=True는 기존 인덱스 2를 일반 데이터 열로 보존하지 않고 버린다는 뜻이다.
# 기존 A, B, C 데이터 열을 삭제한다는 의미가 아니다.
df = df.reset_index(drop=True)
display(df)

# 최종 결과:
#      A    B    C
# 0  7.0  8.0  9.0


# drop=False가 기본값이며, 이 경우 기존 인덱스를 index라는 일반 열로 보존한다.
# 다음과 같이 실행했다면 기존 인덱스 2가 index 열의 데이터로 남는다.
# df = df.reset_index(drop=False)
#
#    index    A    B    C
# 0      2  7.0  8.0  9.0

# 정리:
# drop=True  → 기존 인덱스를 버리고 새 기본 인덱스만 만든다.
# drop=False → 기존 인덱스를 일반 열로 보존하고 새 기본 인덱스를 만든다.

