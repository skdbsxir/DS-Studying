"""
코드 실행 속도 늘리기?
 1. 오버헤드 줄이기 : 모든 연산은 실행하는데 걸리는 오버헤드가 존재. 이런 오버헤드가 쌓이면 전체 성능을 심각하게 저해할 수 있음.
 2. 메모리 관리 : 메모리, 캐시를 적절히 관리.
어떻게 보면 당연한 이야기지만, 간과하기가 쉽다.
CPU는 자기와 가까운 기억장치부터 접근함. 기억장치는 용량이 작을수록 CPU와 가까움.
 > 빠른 연산을 위해서라면 메모리에 저장된 데이터를 최적화하는 것이 중요.
 > well-made된 라이브러리를 선정하고 정확하게 사용해 잘 짜인 프로그램을 만들 수 있다.
"""

# Tip 1. 수치연산 라이브러리 이용.
 ## 각종 수치연산은 넘파이 배열에서 수행하는 것이 훨씬 빠름.
 ## ex. 배열에 있는 모든 값을 +1하는데 걸리는 시간을 비교, 결과 시각화.
  ### 객체에 따라 연산 수행 속도가 다름. 
  ### 일반적인 파이썬 객체는 값을 더하기 전 객체의 자료형을 확인하는 오버헤드가 발생.
  ### 넘파이 배열은 배열의 자료형을 이미 알고있기에 일반적인 객체보다 오버헤드 없이 빠른 연산이 가능.
  ### 또한, 일반적인 파이썬 객체는 넘파이 배열보다 메모리를 더 많이 소모함. 배열이 커지면 배열의 용량이 캐시보다 커져서 더욱 느려지게 됨.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

# 넘파이 배열의 연산시간 측정.
def time_numpy(n) :
    a = np.arange(n)
    start = time.time()
    bigger = a + 1
    stop = time.time()
    return stop - start

# 파이썬 객체의 연산시간 측정.
def time_list(n) :
    l = range(n)
    start = time.time()
    bigger = [x + 1 for x in l]
    stop = time.time()
    return stop - start

n_trials = 10 # 총 시행할 테스트 수
ns = range(20, 30)
ratios = []

for n in ns :
    # 왜 이 2개가 0이 나올까?
    list_total = sum([time_list(n) for _ in range(n_trials)])
    numpy_total = sum([time_numpy(n) for _ in range(n_trials)])
    try :
        ratios.append(list_total / numpy_total)
    except ZeroDivisionError :
        print('Divided by zero.')
        print(list_total)
        print(numpy_total)
        
plt.plot(ns, ratios)
plt.xlabel('Length of array (or list)')
plt.ylabel('$list/numpy$ ratio')
plt.title('Compare the time takes')
plt.show()
"""

# Tip 2. 사용하지 않는 대용량 객체는 바로 삭제.
 ## 파이썬은 가비지 컬렉션을 자동으로 수행하지만, 인터프리터가 항상 완벽하게 객체를 삭제하는건 아님.
 ## del ~~ 을 이용해 사용자가 직접 삭제 가능.
 
# Tip 3. 가능하다면 내장함수 사용.
 ## ex. 직접 더하기 vs. sum()함수 사용
import time

l = range(10000000)
start = time.time()
_ = sum(l) # 내장함수
stop = time.time()
time_fast = stop - start

start = time.time()
sm = 0.0
for x in l : sm += x # 직접 구현
stop = time.time()
time_loop = stop - start

# 데스크탑으론 2.96배 로 나옴.
print('내장함수를 사용하는것이 직접 구현하는 것 보다 %5f배 정도 빠르다.' % (time_loop / time_fast))

# Tip 4. 불필요한 함수 호출은 자제.
 ## 함수는 호출될 때 마다 오버헤드를 발생.
 ## ex. 바로 연산 vs 함수 호출 후 연산
 
add_nums = lambda a, b : a + b # 덧셈 함수 정의 with Lambda표현식
l = range(10000000)

start = time.time()
sm = 0
for x in l : sm += x # 직접 덧셈 수행
stop = time.time()
time_fast = stop - start

start = time.time()
sm = 0
for x in l : sm = add_nums(sm, x) # 함수 이용
stop = time.time()
time_func = stop - start

# 데스크탑으론 1.76배 나옴. --> 함수 호출때문에 발생하는 오버헤드가 전체 실행시간의 절반정도를 차지한 셈.
print('바로 연산하는 것이 함수를 호출하는 것 보다 %5f배 정도 빠르다.' % (time_func / time_fast))

# Tip 5. 덩치가 큰 객체는 가급적 새로 만들지 않는다.
 ## 객체를 갱신할 수 있으면 되도록 갱신. 새로 만드는것은 메모리를 낭비하는 일.
myList = []

# 둘은 결과가 동일. 
myList = myList + [1,2,3] # 방법 1
myList.extend([1,2,3]) # 방법 2
# 하지만, 방법 1은 새로운 리스트를 만들어 myList를 복사하는 과정을 포함함.
# 방법 2는 이미 존재하는 객체인 myList에 새로운 값을 추가하기만 함.
 ## 데이터 용량이 커지면 방법 1은 방법 2보다 속도가 엄청 뒤떨어지는 셈.