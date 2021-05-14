import time
import random
import matplotlib.pyplot as plt
import numpy as np

# 퀵 정렬
def quickSort(myList) :
    if len(myList) <= 1:
        return myList
    pivot = random.choice(myList)
    
    lesser = [x for x in myList if x < pivot]
    greater = [x for x in myList if x > pivot]

    sortedLesser = quickSort(lesser)
    sortedGreater = quickSort(greater)
    
    return sortedLesser + [pivot] + sortedGreater

def merge(list1, list2) :
    newList = []
    while len(list1) > 0 or len(list2) > 0 :
        if len(list1) > 0 and len(list2) > 0 :
            if list1[0] <= list2[0] :
                newList.append(list1[0])
                list1 = list1[1:]
            else :
                newList.append(list2[0])
                list2 = list2[1:]
        elif len(list1) > 0 :
            newList.append(list1[0])
            list1 = list1[1:]
        elif len(list2) > 0 :
            newList.append(list2[0])
            list2 = list2[1:]
    return newList

# 병합 정렬
def mergeSort(myList) :
    if len(myList) <= 1 :
        return myList
    
    mid = len(myList) // 2
    L1 = myList[:mid]
    L2 = myList[mid:]
    S1 = mergeSort(L1)
    S2 = mergeSort(L2)
    
    return merge(S1, S2)

# 버블 정렬
def bubbleSort(myList) :
    for i in range(len(myList)) :
        for j in range(len(myList)) :
            if myList[i] < myList[j] :
                myList[i], myList[j] = myList[j], myList[i]
    return myList

# 삽입 정렬
def insertionSort(myList):
    for end in range(1, len(myList)) :
        for i in range(end, 0, -1) :
            if myList[i - 1] > myList[i] :
                myList[i - 1], myList[i] = myList[i], myList[i - 1]
    return myList
    
# 시간 재기
def timeCounter(func, arg) :
    start = time.time()
    func(arg)
    stop = time.time()
    return stop - start

times_Quick, times_Merge, times_Bubble, times_Insertion = [], [], [], []

ns = range(1000) # 리스트 길이 지정
for i in ns :
    randomList = list(np.random.uniform(size=i))
    times_Quick.append(timeCounter(quickSort, randomList))
    times_Merge.append(timeCounter(mergeSort, randomList))
    times_Bubble.append(timeCounter(bubbleSort, randomList))
    times_Insertion.append(timeCounter(insertionSort, randomList))
    
plt.plot(times_Quick, '--', label = 'QuickSort $O(n*logn)$')
plt.plot(times_Merge, '-.', label = 'MergeSort $O(n*logn)$')
plt.plot(times_Bubble, label = 'BubbleSort $O(n^2)$')
plt.plot(times_Insertion, ':', label = 'InsertionSort $O(n^2)$')
plt.xlabel('Length of list')
plt.ylabel('Time (sec)')
plt.title('<Time to sorting given list>')
plt.legend(loc='upper left')
plt.show()