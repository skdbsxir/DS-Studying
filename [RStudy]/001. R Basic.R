age <- 20  # 값 대입.
age

age <- 30 # 값 덮어쓰기.
age

a = 20 # 대입? 이거도 되네.
a

class(age) # type 확인. numeric.

name = 'KCH'
class(name) # character.

is_effective = TRUE # boolean. TRUE 대신 T 또한 가능.
is_effective

is_effective2 = FALSE
is_effective2

class(is_effective)

# Factor Type. 범주형 저장을 위한 type.
sido = factor('서울',c('서울','부산','제주') )
sido

class(sido)
levels(sido)

null1 = NULL # 초기화 할때 주로 사용.
score = c(NA, 90, 100) # NA 결측치

10/0
0/0