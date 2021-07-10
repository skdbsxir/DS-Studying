no = c(10, 20, 30, 40, 50, 60, 70)
age = c(18, 15, 13, 12, 10, 9, 7)
gender = c('M', 'M', 'M', 'M', 'M', 'F', 'M')

students = data.frame(no, age, gender) # DataFrame 생성
students

colnames(students) # 열 이름 확인
rownames(students) # 행 이름 확인

colnames(students) = c('no', '나이', '성별') # 열 이름 변경
rownames(students) = c('A', 'B', 'C', 'D', 'E', 'F', 'G') # 행 이름 변경
students

colnames(students) = c('no', 'age', 'gender') # 재변경

students$no # $를 이용해 특정 col 접근.
students$age
students[,'no'] # 대괄호 써서 접근도 가능.
students[,'age']
students[, 1] # col 이름이 아닌 col index로도 접근 가능.
students[, 2]

students['A',] # 특정 row 접근. 행이름 이용.
students[2, ] # 특정 row 접근. row index 이용.

students[3,1] # 특정 데이터 접근. 3행1열 접근.
students['A', 'no'] # A행의 no 열 접근.

students$name = c('이용', '준희', '이훈', '서희', '승희', '하정', '하준') # 열 추가.
students

students['H', ] = c(80,10,'M','길동') # 행 추가
tail(students)

var1 = c(1:12)
arr1 = array(var1, dim=c(2,2,3)) # 3차원 배열 생성 (2행 2열 이 3개 면으로 생성)
arr1
arr2 = array(var1, dim=c(12)) # 1차원 배열
arr2
arr3 = array(var1, dim=c(6,2)) # 2차원 배열
arr3
arr4 = array(var1, dim=c(2,2,3,1)) # 4차원 배열
arr4

v_data = c('02-111-2222', '01022223333') # 벡터
m_data = matrix(c(21:26), nrow=2) # 행렬
a_data = array(c(31:36), dim=c(2,2,2)) #3차원배열
d_data = data.frame(address=c('seoul', 'busan'), names=c('Lee', 'Kim'), stringsAsFactors=FALSE) # 데이터프레임
list_data = list(name='길동', tel=v_data, score1=m_data, score2=a_data, friends=d_data) # 각 key에 value 지정.
list_data

# 리스트는 다차원 데이터 저장구조.
list_data$name # name 키와 쌍을 이루는 데이터 추출
list_data$tel # tel 키와 쌍을 이루는 데이터 추출
list_data[1] # 첫번째 서브 리스트추출
