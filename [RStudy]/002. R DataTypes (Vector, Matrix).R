# 비교연산은 타 언어와 동일.
10 <= 10
10 > 5
10 >= 5

# Python의 if ~~ in list 같은거. 있으면 TRUE.
n = 20
n %in% (c(10, 20, 30))

# and, or는 1개만.
n1 = 10
n1 >= 0 & n1 <= 100
n2 = 1000
n2 >= 0 & n2 <= 100

# not연산자. 동일.
!(10==10)


students_age = c(11, 12, 13, 20, 15, 21) # Vector. 같은 data type으로만.
students_age
class(students_age)
length(students_age)

students_age[1]
students_age[3]
students_age[-1]

students_age[1:3]
students_age[4:6]

score = c(1,2,3)
score[1] = 10 # 1 index값을 10으로.
score[4] = 4 # 4 index 추가, 4 삽입.
score

code = c(1,12,'30')
class(code)
code

data = c(1:10) # 순열
data

data1 = seq(1,10) # 1~10까지 1씩 증가.
data1

data2 = seq(1, 10, by=2) # 1~10까지 2씩 증가.
data2

data3 = rep(1, times=5) # 1을 5번.
data3

data4 = rep(1:3, each=3) # 1~3을 각각 3번씩.
data4

var1 = c(1,2,3,4,5,6)
x1 = matrix(var1, nrow=2, ncol=3) # var1을 써서 2*3 행렬 생성. 기본적으로 열 우선으로 값 생성.
x1
x2 = matrix(var1, ncol=2) # var1을 써서 2열 행렬 생성. row는 자동 계산됨. 
x2

x1[1,] # x1의 1행, 모든 열 선택
x1[,1] # x1의 모든 행, 1열 선택
x1[2,2] # x1의 2행 2열 선택.

x1 = rbind(x1, c(10,10,10)) # x1에 행 추가
x1 = cbind(x1, c(20,20,20)) # x1에 열 추가
x1