# Orange는 내장 dataset.
head(Orange) # 6번째 까지 추출.
head(Orange, 3) # 3번째 까지 추출.

tail(Orange) # 마지막부터 6번째까지.
tail(Orange, 3) # 마지막부터 3번째 까지.

str(Orange) # 데이터 구조 파악.

summary(Orange) # Python의 DataFrame.describe() 같은 거. 기술통계량 반환.
# 범주형의 경우 빈도수 반환.

# read.csv 써서 csv 읽어올 수 있음.
nhis = read.csv('data/NHIS_OPEN_GJ_EUC-KR.csv')
head(nhis)

# 인코딩 문제로 잘 안불러질때도 있음. fileEncoding에 인자 넣으면 OK.
# 보통은 EUC-KR or UTF-8
nhis = read.csv('data/NHIS_OPEN_GJ_EUC-KR.csv', fileEncoding = 'EUC-KR')
head(nhis)

# 만일 열 이름이 없는 (헤더없는) 파일을 읽는다면?
# header = FALSE 지정.
# 문자열 데이터를 범주형으로 읽고자 한다면 stringAsFactor = TRUE 지정.
sample = read.csv('data/sample.csv', header=FALSE, fileEncoding='EUC-KR', stringsAsFactor=TRUE)
sample
str(sample)

# 엑셀파일 읽기? openxlsx 패키지의 read.xlsx() 사용.
install.packages('openxlsx')
library(openxlsx)
sheet1 = read.xlsx('data/NHIS_OPEN_GJ_EUC-KR.xlsx') # default는 첫번째 시트 읽음.
sheet2 = read.xlsx('data/NHIS_OPEN_GJ_EUC-KR.xlsx', sheet=2) # sheet인자를 통해 시트 지정 가능.
head(sheet1)
head(sheet2)

# 큰 용량 데이터 파일 읽기
# data.table 패키지의 fread() 이용. read.csv보다 읽는 속도가 빠름.
install.packages('data.table')
library(data.table)
bigdata = fread('data/NHIS_OPEN_GJ_BIGDATA_UTF-8.csv', encoding='UTF-8')
str(bigdata)

###########################################
###########################################

# 행 인덱스 써서 데이터 추출?
Orange[1,] # 1행만 추출
Orange[1:5,] # 1~5행 추출
Orange[6:10,] # 6~10행 추출
Orange[c(1,5),] # 1, 5행만 추출
Orange[-c(1:29),] # 1~29행 제외하고 추출

# 조건식을 통해 추출할 수도 있음.
Orange[Orange$age == 118, ] # age 컬럼의 데이터가 118인 행만 추출.
Orange[Orange$age %in% c(118, 484), ] # age 컬럼의 데이터가 118 or 484인 행만 추출.
Orange[Orange$age >= 1372, ] # age 컬럼의 데이터가 1372 이상인 행만 추출.

# 컬럼명을 통한 추출?
Orange[, 'circumference']

# 특정 열만 추출, 행은 1행만 추출.
Orange[1, c('Tree', 'circumference')]

# 마찬가지로 인덱스 써서 가져올 수 있음.
Orange[, 1] # 1열 추출
Orange[, c(1,3)] # 1열, 3열 추출
Orange[, c(1:3)] # 1~3열 추출
Orange[, -c(1,3)] # 1,3열 제외하고 추출

# 행,열 모두 조건?
Orange[1:5, 'circumference'] # 1~5행, circumference 열만 추출.
Orange[Orange$Tree %in% c(3,2), c('Tree', 'circumference')] # Tree열이 3 or 2인 행의 Tree 열과 circumference 열 추출


# 정렬해서 출력? order() 사용.
OrangeT1 = Orange[Orange$circumference < 50, ] # circumference가 50 미만인 행만 추출
OrangeT1[order(OrangeT1$circumference), ] # 이를 오름차순으로 정렬해서 출력.
OrangeT1[order(-OrangeT1$circumference), ] # 내림차순은 앞에 - 붙이면 됨.

# 그룹별 집계? aggregate() 사용.
 # aggregate(계산할속성 ~ 대상속성, 데이터, 계산할값)
# Tree 컬럼의 값이 같은 행 끼리 묶고, circumference의 평균 계산.
 # 일종의 그룹끼리 묶어서 평균 계산한 것.
aggregate(circumference ~ Tree, Orange, mean)

###########################################
###########################################

# 데이터 병합? merge() or cbind() 사용.
stu1 = data.frame(no=c(1,2,3), midterm=c(100,90,80))
stu2 = data.frame(no=c(1,2,3), finalterm=c(100,90,80))
stu3 = data.frame(no=c(1,4,5), quiz=c(99,88,77))
stu4 = data.frame(no=c(4,5,6), midterm=c(99,88,77))

# merge()는 동일 컬럼명의 동일 데이터 행 끼리 병합.
stu1
stu2
merge(stu1, stu2)
stu3
merge(stu1, stu3) # 동일 컬럼명의 동일 데이터 행이 없으면 합치지 않음.
merge(stu1, stu3, all=TRUE) # all=TRUE 지정시 다 합침. Full Outer Join하는 셈.

# rbind()는 행 방향으로 병합. 단, 두 데이터프레임의 컬럼명이 동일해야 함.
stu1
stu4
rbind(stu1, stu4)

# cbind()는 열 방향으로 병합. 
stu1
stu2
cbind(stu1, stu2) # merge()와는 다르게 말 그대로 옆으로 붙음.
merge(stu1, stu2)