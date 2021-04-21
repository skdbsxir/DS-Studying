import pandas as pd # 팬더스 라이브러리 호출, pd로서 사용. (numpy as np 처럼)

"""
# 간단한 데이터 프레임은 pd.DataFrame으로 사용
# 내부 구조는 딕셔너리로 구성. (딕셔너리 값으로 데이터 프레임을 구성)
# 즉, 키 값은 열 이름이 되고 각 키에 해당하는 값들이 행 성분이 된다.
# 따라서, 현재 생성한 데이터 프레임은 3x2(3행2열) 크기의 데이터 프레임이 된다.
# 생성된 데이터 프레임이 보고 싶으면? print(df)하면 됨.
df = pd.DataFrame({
        "name" : ["Bob", "Alex", "Janice"],
        "age" : [60, 25, 33]
        })

# 이런 식으로 csv 파일을 불러와서 데이터 프레임의 생성이 가능.
#other_df = pd.read_csv("myfile.csv")


#print(df.index)
#print(df.pivot_table)
# 기존 데이터 프레임에 있는 열을 사용해 새로운 열을 추가할 수 있다.
df["age_plus_one"] = df["age"] + 1
df["age_times_two"] = 2 * df["age"]
df["over_30"] = (df["age"] > 30) # 결과 값은 True || False

# 데이터 프레임 열에 대해 내장 통계함수를 사용할 수 있다.
total_age = df["age"].sum() #전체 합
median_age = df["age"].quantile(0.5) # 중간 값

# 데이터 프레임에서 여러 행을 불러와서 또 다른 데이터 프레임을 만들 수 있다.
df_below50 = df[df["age"] < 50]

#print(df.index)
#print(df.pivot_table)
#print(median_age)
#print(df)


#위와 내용은 같지만, index로 name을 사용하는 데이터 프레임을 생성.
df_w_name_as_ind = df.set_index("name")
print(df_w_name_as_ind.index) # index로 사용하는 값들을 출력. 출력은 Bob, Alex, Janice가 된다.

# 위에서 name을 index로 사용하는 데이터 프레임에서 Bob행의 값을 읽는다.
bobs_row = df_w_name_as_ind.loc["Bob"]
print(bobs_row)

# 특정 값 만을 읽고 싶으면? 참조하면 됨.
print(bobs_row["age"])

"""
"""
# 시리즈 == 데이터 프레임의 열 : 동일한 자료형으로 구성
s = pd.Series([1,2,3]) # 크기 3의 리스트를 이용해 시리즈를 만든다.
print(s)
print(s+2) # 시리즈의 각 성분에 +2 한 결과를 출력
print(s.index)

# 길이가 같은 두 시리즈를 더하면 성분별로 덧셈을 한다. 
# int형 + float형 하면 결과는 float형으로 출력. (한 시리즈에 float형이 있으면 나머지 int형도 전부 float형으로 바뀜)
t = s + pd.Series([4,5,6]) 
print(t)
"""
# JOIN과 GROUP연산 사용
 # JOIN은 두 데이터 프레임을 합칠때, GROUP은 한 데이터 프레임을 나눌때.
df_w_age = pd.DataFrame({
        "name" : ["Tom", "Tyrell", "Claire"],
        "age" : [60, 25, 33]
        })

df_w_height = pd.DataFrame({
        "name" : ["Tom", "Tyrell", "Claire"],
        "height" : [6.2, 4.0, 5.5]
        })

# 두 데이터 프레임을 name을 index로 하여 join한다. (df_w_age JOIN df_w_height on "name" 인 셈.) (name = pk)
joined = df_w_age.set_index("name").join(df_w_height.set_index("name"))

#print(joined)
#print(joined.reset_index)

# 간단한 데이터 프레임을 정의
df = pd.DataFrame({
        "name" : ["Tom", "Tyrell", "Claire"],
        "age" : [60, 25, 33],
        "height" : [6.2, 4.0, 5.5],
        "gender" : ["M", "M", "F"]
        })
#print(df)
# 생성한 데이터 프레임을 gender를 index로 하여 grouping 한다. 
 # groupby() 함수의 절차는 split-apply-combine. 그룹분석 함수.
 # 즉, 기준 열을 지정 -> 특정 열을 그룹 별로 분할 -> 각 그룹에 통계함수 적용 -> 최종적으로 산출된 통계량을 통합해서 표시해주기 때문.
 # 그냥 통계량 구하기 위해서 쓰는 함수라고 봐도 될듯.
gender_grouping = df.groupby("gender")
print(gender_grouping.mean()) # 그냥 gender_grouping만 하면 그루핑 되었다는 결과만 보여줌. 뒤에 통계함수를 붙여야 통계함수가 적용 된 결과를 산출.

# 이번엔 평균값이 아닌 중앙값으로.
medians = gender_grouping.quantile(0.5)

# 위 데이터 프레임에 대해 연산 agg를 각 열마다 적용해보자. 결과 출력물은 시리즈. 데이터 프레임으로 한다면? pd.DataFrame으로 하면 될 것.
 # 안대네.. ValueError: If using all scalar values, you must pass an index
 # 중괄호 뒤에 , index = [0] 해주면 된다.
def agg(ddf):
    return pd.Series({
            "name" : max(ddf["name"]),
            "oldest" : max(ddf["age"]),
            "mean_height" : ddf["height"].mean()
    })
    
print(gender_grouping.apply(agg))







