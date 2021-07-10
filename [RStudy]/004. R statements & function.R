score = c(10, 20)
score + 2 # 각 모든 값에 2를 더함.
score # score 자체는 변하지 않음.

score = score + 2 # 결과를 반영하려면 계산한 값을 기존 값에 넣어줘야 함. 원래 하던것처럼.
score

score1 = c(100, 200)
score2 = c(90, 91)
score3 = c(100, 200, 300, 400)

# 벡터 간 연산은 같은 위치끼리만.
sum_score = score1 + score2
sum_score # 100+90 // 200+91 
diff_score = score1 - score2
diff_score # 100-90 // 200-91

# 갯수가 안맞는경우는?
# 다른쪽이 적은쪽의 배수 갯수만큼 크다면 경고문구 X. 결과가 나오긴 함.
score2 + score3

m1 = matrix(c(1:6), nrow=2)
m1 = m1 * 10 # 각 행렬 원소에 *10 연산.
m1

m1 = matrix(c(1:9), nrow=3)
m2 = matrix(rep(2, times=9), nrow=3)
m1
m2
m1 + m2 # 행렬 간 연산은 같은 위치끼리만.

#################

score = 95

# if 문은 동일
if (score >= 80) {
  print('조건이 True이므로 수행.')
}

# if-else는 괄호 뒤에 붙여서.
if (score >= 100) {
  print('A+')
} else{
  print('A or B or C or D')
}

# if-else if도 마찬가지로 괄호 뒤에 붙여서.
if (score >= 100){
  print('A+')
} else if (score >= 91){
  print('A')
} else if (score >= 81){
  print('B')
} else if (score >= 71){
  print('C')
} else if (score >= 61){
  print('D')
} else{
  print('F')
}

# ifelse(조건, '조건이 TRUE일때 수행할 문장', '조건이 FALSE일때 수행할 문장') 형태.
score2 = 85
ifelse(score2>=91, 'A', 'FALSE이므로 B')

# for 문은 형태 비슷.
# (1:3) -> 1부터 3까지 1씩 증가시킴. 
# 순서대로 num이 이를 참조하며 출력.
for (num in (1:3)){
  print(num)
}

# 마찬가지로 if를 넣어 제어가 가능.
# paste는 괄호 안 내용들을 붙여서 문자열로 만들어줌.
for (num in (1:5)){
  if (num %% 2 == 0){
    print(paste(num, '짝수'))
  } else{
    print(paste(num, '홀수'))
  }
}

################################

# 함수 생성? 변수 선언처럼 함수로 생성하면 됨.
a = function(){
  print('testa')
  print('testA')
}
a() # 호출은 변수 뒤에 ()를 붙여서.

# 인자 있는 함수? 인자만 넣으면 됨.
a = function(num){
  print(num)
}
a(20)
a(10)

a = function(num1, num2){
  print(paste(num1, '', num2))
}
a(10, 20)
a(num1=10, num2=20)
a(num2=20, num1=10) # 매개변수를 지정해서 넘기면 알아서 매핑해줌.

# 반환값 있는 함수?
calculator = function(num1, op, num2){
  result = 0
  if (op == '+'){
    result = num1 + num2
  } else if (op == '-'){
    result = num1 - num2
  } else if (op == '*'){
    result = num1 * num2
  } else if (op == '/'){
    result = num1 / num2
  }
  return (result) # 반환값을 괄호로 묶어줌.
}
calculator(1, '+', 2)
calculator(1, '-', 2)