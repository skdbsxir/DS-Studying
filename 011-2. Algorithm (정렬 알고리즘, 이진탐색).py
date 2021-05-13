# [1] 길이가 n인 배열에서 원소 x의 위치를 찾는 문제.
L = list(map(int, input().split())) # 1, 2, ... , n
x = int(input())

def foundNumber(X, myList) :
    for i in range(len(myList)) :
        # 이 작업의 복잡도는 O(1).
        # 이를 전체 리스트 길이만큼 for문을 돌리므로 최대 n번, 평균 n/2번 수행하므로 복잡도는 O(n)이 된다.
        if myList[i] == X :
            return True
    return False
print(foundNumber(x, L))

# [3] 배열을 정렬하는데 걸리는 시간을 어떻게 될까?
# 2번의 반복문을 통해 리스트를 정렬. O(n^2)의 시간이 걸린다.
# Bubble Sort. 시간이 오래 걸리지만, 구현이 간단한 장점이 있음.
def simpleSort(myList) :
    for i in range(len(myList)) :
        for j in range(len(myList)) :
            if myList[i] < myList[j] :
                myList[i], myList[j] = myList[j], myList[i]
    return myList

# [4] 위의 Bubble sort는 상당히 느림. 리스트의 길이가 커진다면 시간이 상당히 오래 걸릴 것.
# 다른 접근방법? 주어진 리스트를 반으로 쪼갠 후, 각각을 정렬한 다음 합친다. 이를 반복.
# Merge Sort.
def mergeSort(myList) :
    if len(myList) <= 1 :
        return myList
    mid = len(myList) // 2
    # 길이가 n인 리스트를 n/2로 나눠서 재귀적으로 푼다.
    # 이진탐색과 마찬가지로 배열을 총 log(n)번 나누게 되고, 매 과정마다 O(n)의 복잡도가 있는 merge를 수행.
    # 따라서, 최종 복잡도는 O(nlogn)이 된다.
    L1 = myList[:mid]
    L2 = myList[mid:]
    S1 = mergeSort(L1)
    S2 = mergeSort(L2)
    return merge(S1, S2)

# 리스트를 병합하는 merge 함수.
def merge(list1, list2) :
    newList = []
    # 복잡도? 비교횟수에 비례. O(n)
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
    """
    while not list1 and not list2 :
        if(list1[0] <= list2[0]) or not list2 :
            mn = list1[0]
            del list1[0]
        else :
            mn = list2[0]
            del list2[0]
        newList.append(mn)
    if not list1 :
        newList.extend(list1)
    elif not list2 :
        newList.extend(list2)
    return newList
    """
print(mergeSort(L))

# [2] 만일, 주어진 리스트가 '정렬된 자료'라면 좀 더 빠르게 찾을 수 있다.
# 길이가 n인 문제를 n/2인 문제로 변환. 배열의 중간값을 x와 비교하고,
# 중간값이 x보다 크다면 왼쪽부분만을 선택, 왼쪽부분의 중간값을 다시 x와 비교.
# 이를 이진탐색 이라고 한다. (i = 0, j = n - 1)
def binarySearch(X, myList) :
    i = 0
    j = len(myList)-1 # 리스트의 길이 - 1
    # myList.sort() # 미리 정렬해둔다. (내장 sort 함수. Tim sort. O(nlogn))
    # simpleSort(myList) # Bubble Sort. O(n^2)
    mergeSort(myList) # Merge Sort. O(nlogn)
    while True :
        mid = myList[(i + j) // 2] # 중간값 지정. 해당 변환 작업은 O(1).
        if X == mid :
            return True
        elif j == i or j == i+1 :
            return False
        else :
            if mid > X :
                j = (i + j) // 2
            else :
                i = (i + j) // 2
    # 이를 계속 반복하면 O(1)의 복잡도가 추가. 
    # 길이가 n이던 리스트가 반으로 나뉘면서 점점 짧아지고 결국 1이되면 반복이 종료.
    # 이는 총 log(n)번 발생. --> 이진탐색의 복잡도는 O(log(n))이 된다. (n이 크면 O(n)과 비교해 엄청나게 빠름.)
print(binarySearch(x, L))