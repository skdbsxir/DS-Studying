"""
중복으로 나타나는 원소가 몇개있는지 count? 2가지 방법.
1. 각 원소에 대해 전체 배열에서 몇번 나타나는지 count. 모든 원소에 대해 배열을 한 번씩 훑는다.
2. 배열을 훑으면서 key=원소, value=값 으로 같는 딕셔너리를 채운 후 값이 1보다 큰 키가 몇개있는지 찾는다.
"""
import time
import matplotlib.pyplot as plt
import numpy as np

# O(n^2) 시간
def duplicates_On2(myList) :
    counter = 0
    # 리스트 전체를 읽으며 모든 값을 count.
    for i in myList :
        if myList.count(i) > 1 :
            counter += 1
    return counter

# O(n) 시간
def duplicates_On(myList) :
    counter = {} # 딕셔너리 이용.
    for i in myList :
        # 원소가 리스트에 있으면 값 +1
        if i in counter :
            counter[i] += 1
        # 없으면 1
        else :
            counter[i] = 1
    counts_above_1 = [ct for i, ct in counter.items() if ct > 1]
    return sum(counts_above_1)

# 함수의 수행 시간을 재는 함수. 
def timeit(func, arg) :
    start = time.time()
    func(arg)
    stop = time.time()
    return stop - start

times_On, times_On2 = [], []

ns = range(1000) 
for i in ns :
    # 임의의 리스트 생성.
    countList = list(np.random.uniform(size=i))
    # 생성한 리스트에서 두 함수를 이용해 중복값을 찾는다.
    times_On.append(timeit(duplicates_On, countList))
    times_On2.append(timeit(duplicates_On2, countList))
    
plt.plot(times_On2, '--', label='$O(n^2)$')
plt.plot(times_On, label='$O(n)$')
plt.xlabel('Length of list')
plt.ylabel('Time (sec)')
plt.title('<Time to count duplicate numbers>')
plt.legend(loc='upper left')
plt.show()

# 교재랑 다르게 내껀 뭔가 지저분하게 나온다? 리스트 크기가 작아서 그랬는듯.
# 리스트 크기가 늘어날수록 O(n)이 더 빠르게 수행됨을 볼 수있다.