# 빅오 복잡도는 최악의 경우를 가정한 복잡도.
# 최악의 경우가 드물게 발생한다면 실제 현실에서 발생하는 수행 시간과의 괴리가 생기게 된다.
# 그래서 보통 평균복잡도 를 고려함.

# Quick Sort.
# 주어진 배열에서 임의의 기준을 잡고, 임의 순서대로 섞는 작업을 먼저 수행하므로 최악의 경우가 발생하는 일이 드물다.
# 종종 병합정렬 보다 빠르게 작동함.

import random # 임의 추출을 위해 import.
L = list(map(int, input().split())) # 1, 2, ... , n
# x = int(input())

def quickSort(myList) :
    if len(myList) <= 1 :
        return myList
    elem = random.choice(myList) # 기준점(pivot) 설정
    
    # 기준점을 기준으로 리스트에서 이보다 작거나 큰 값들을 좌/우로 나눈다.
    # 그렇게 되면 기준점은 일단 정렬된 위치에 존재.
    lessthan = [x for x in myList if x < elem]
    morethan = [x for x in myList if x > elem]
    
    # 정렬되지 않은 좌/우를 동일한 방법으로 나눈다.
    # 이를 반복하면 정렬된 기준점을 기준으로 좌/우가 나뉘고, 좌/우가 나뉘고...
    # 결국 첫번째 기준점을 기준으로 좌, 우 모두 정렬된 상태가 된다. 으앙 유레카 인복이형!!!!!!!!!
    sortless = quickSort(lessthan)
    sortmore = quickSort(morethan)
    
    return sortless + [elem] + sortmore

# 퀵 정렬은 임의로 선택한 기준점이 항상 배열에서 가장 큰값이라면 O(n^2)의 복잡도를 가지지만, 그런 일은 거의 일어나지 않는다.
# 평균적으로 O(nlogn)의 복잡도를 가짐.
print(quickSort(L))


# 분할상환분석(Amortized Analysis)은 특정 연산의 복잡도를 측정하는 방법. 평균 복잡도와 유사하지만 큰 차이점이 있음.
# 평균 복잡도는 평균적으로 그 정도의 복잡도를 갖는다는 것을 의미. 실제 연산에서 최악의 경우를 보장하진 않음.
# 분할상환분석은 알고리즘 전체의 복잡도가 아닌, 어떤 자료구조에서 특정 연산을 수행하는데 걸리는 시간을 측정.
 ## 이 시간은 운이 좋든, 안좋든, 어떤 데이터를 어떻게 사용하든 간에 평균적으로 시간이 얼마만큼 걸리는지를 정확히 보장함.

# 딕셔너리를 예로 생각해보자.
 ## 딕서녀러에 새 원소를 추가하는 연산의 복잡도는 보통 O(1).
 ## 딕셔너리의 크기가 커지다 보면 가끔 O(n)의 시간이 걸리는 작업을 수행하기도 함.
 ## 하지만 이 작업은 임의의 순간에 발생하는 것이 X. 발생 횟수 또한 제한되어 있음.
 ## 따라서 어떤 경우에도 딕셔너리에 원소를 추가하는 연산은 평균 O(1)이 걸린다는 것을 보장할 수 있다.
 
# 실제 값을 추가해보며 시간이 얼마나 걸리는지 측정. O(n)연산은 아주 드물게 수행될 뿐 만 아니라, 언제 몇번 일어나는지 정해져 있음.
import time
import matplotlib.pyplot as plt

times, myDict = [], {}
for i in range(10000000) :
    start = time.time()
    myDict[i] = i
    stop = time.time()
    times.append(stop - start)
    
plt.plot(times)
plt.xlabel('Size of dictionary')
plt.ylabel('Time to adding values (sec)')
plt.show()
 