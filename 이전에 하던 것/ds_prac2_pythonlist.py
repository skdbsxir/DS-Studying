"""
#단순 리스트 조작
list1 = ["a", " b", " c"]
print(list1)

list1[0] = "A" #리스트 맨 앞 a를 A로 
print(list1)

list1.append('d')  #리스트 마지막에 d를 추가
print(list1)

print(list1[0:2]) #리스트 의 두번째 원소까지 출력

print(list1[0:]) #좌측에 지정된 위치부터 리스트의 끝까지 출력

print(list1[:4]) #리스트의 처음부터 우측에 지정된 위치까지 출력

print(list1[:-1]) #리스트의 처음부터 리스트의 마지막-1 위치까지 출력
"""

"""
#문자열 리스트 조작
list2 = "I am a boy".split() # 작성한 리스트를 공백 기준으로 여러개의 문자열로 나눈다.
print(list2)

list3 = "I am a boy".split("a") # 괄호 안의 문자를 제외, 공백을 기준으로 여러개의 문자열로 나눈다.
print(list3)

print(" Haha ".join(["A", "B", "C"])) # join 괄호 안에 있는 문자들 사이에 해당 리스트를 추가한다.

print(list2[0: 4: 2]) # list2의 첫번째 인덱스 ~ 4번째 인덱스 까지 2칸 간격으로 slice한다.
print("0123456789"[0: 10: 3])
"""

"""
#튜플
tuple1 = (1, 2, "ABC") # 튜플 선언.
print(tuple1[0]) # 튜플의 첫번째 원소값을 출력
tuple1[0] = 5 # 에러발생. 튜플은 선언되고 나면 수정 불가능한 속성을 갖는다.
zero, one, two = tuple1 # 튜플의 성분을 3개의 별도의 변수에 할당한다. 
print(two) # two 변수를 출력. 그러면 tuple안의 3번째 내용인 ABC가 출력된다.
zero = 3
print(zero) # 이렇게 하면 에러발생 X. 튜플이 아닌 단순히 변수를 수정하는 것이기 때문.
print(tuple1) # 실제로 출력하면 튜플은 수정되지 않았음. 너무나도 자명하다
"""

"""
#딕셔너리 (자바에서 한 HashTable과 유사?)
  # 유사하지만 차이점이 있음. 딕셔너리는 generic type, 해시테이블은 non-generic type.
  # 해시 테이블은 weakly typed data structure, 딕셔너리는 strongly typed data structure.
  # 해시 테이블을 이용해 딕셔너리 구현이 가능. (레드-블랙 트리를 이용해서도 구현이 가능.)

mydict = {"Jan" : 1, "Feb" : 2} # 딕셔너리의 선언. {키1 : 값1, 키2 : 값2, ...}과 같은 형식으로 선언한다.
print(mydict["Jan"]) # 딕셔너리 Jan의 값을 출력. 1이 출력된다.
mydict["Mar"] = 3 # 딕셔너리에 Mar를 추가. Mar의 속성값은 3으로 지정.
print(mydict) # 딕셔너리에 키(Mar)와 값(3)이 추가된 것을 확인할 수 있다.
mydict["Jan"] = "1월" # 딕셔너리의 키 Jan의 값을 1월 로 변경한다.
print(mydict) # 1->1월 로 변경되어 있는 것을 확인할 수 있다.
dict_to_list = mydict.items() # 딕셔너리를 (키, 값)의 쌍으로 이루어진 리스트로 분할한다.
print(dict_to_list) # 리스트로 분할된 모습을 확인할 수 있다. 근데 앞에 dict_items는 왜있는거지
list_to_dict = dict(dict_to_list) # (키, 값)의 쌍으로 이루어진 리스트를 딕셔너리로 변환한다.
print(list_to_dict) # 다시 딕셔너리로 변환된 모습을 확인할 수 있다.
print("Jan" in mydict) # in을 이용해 딕셔너리에 해당하는 키가 있는지 확인 가능.
"""

"""
# 집합 (set) - 딕셔너리에서 값을 제외한 자료구조로 생각해도 됨.
  # 순서 유지 X, 객체의 중복을 허용하지 않는 컨테이너.
myset = set() # 공집합 생성
print(myset) # 집합을 출력.
print(1 in myset) # 집합에 1이 있는지? 현재는 없으므로 False를 출력.
myset.add(1) # 집합에 1을 추가.
print(myset) # 집합을 출력. 1이 추가된 걸 확인할 수 있다.
print(1 in myset) # 집합에 1이 있는지? 현재는 있으므로 True를 출력.
myset.add(1) # 집합에 또 1을 추가?
print(myset) # 아무런 일도 일어나지 않는다. 여전히 집합은 {1}인 상태.
"""

"""
# 함수
def sqrt_func(x):
    x_sqrt = x * x
    return x_sqrt

chkval = sqrt_func(2)
print(chkval)


def func1(x, n=2): # 함수 인자의 초기값을 다음처럼 default 값으로 설정할 수 있다.
    return pow(x, n)

chkval2 = func1(2) # 함수 호출 시 default값에 인자를 넘겨주지 않아도 default값이 함수에 전달되는 셈.
print(chkval2)
chkval3 = func1(2, 3) # default로 설정된 값 외에 다른 값을 넣어도 OK.
print(chkval3)


def func2(x):
    x = x + 1
    print(x) # 여기선 2 + 1 = 3 이 출력되긴 함.

chkval4 = func2(2)
print(chkval4) # 함수에서 return값이 없으므로 함수의 출력은 None이 된다.


sqr = lambda x : x * x # 간단한 함수는 다음처럼 lambda 표현식으로도 정의가 가능. (lambda는 Lisp에서 함수를 정의할 때 사용하던 이름.)
chkval5 = sqr(5) # 함수의 이름이 sqr인 셈.
print(chkval5)


def apply_to_evens(a_list, a_func):
    return [a_func(x) for x in a_list if x % 2 == 0]

my_list = [1,2,3,4,5]
sqrs_of_evens = apply_to_evens(my_list, lambda x : x * x) # 간단한 람다 함수는 인자로 전달할 수도 있음. 이런 람다함수를 익명 함수라고 함.
print(sqrs_of_evens)
"""

"""
# 반복문, 제어문
  # 간단한 반복문. x는 반복자.
my_list = [1,2,3]
for x in my_list:
    print("the number is ", x)
 
  # 딕셔너리를 반복문에 적용. 반복문에 사용가능한 자료구조를 iterable한 자료구조라 한다.
mydict = {"Jan" : 1, "Feb" : 2}
for key, value in mydict.items():
    print("the value for ", key, " is", value)

  # 간단한 제어문. if-elif-else로 이루어짐. else if가 아니라 elif임을 주의.
i = 5
if i < 3:
    print("i는 3미만")
elif i < 5:
    print("i는 3이상 5미만")
else:
    print("i는 5이상")

  # 간단한 while문. 
j = 0
while j < 5:
    print("j는 5보다 작다")
    j += 1
"""
"""
# 그 외 함수들

print(int(5.7)) # 실수 -> 정수
print(float(5)) # 정수 -> 실수
print(bool("")) # 아무런 문자도 없으면 False
print(bool("asddg")) # 문자가 있는경우 True
print(str(5)) # 문자열로 변환
print(dict([("Jan", 1), ("Feb", 2)])) # 튜플로 구성된 리스트를 딕셔너리로 변환
"""
"""
# 예외처리
try:
    line = input_text.split("\n")
    print("asd")
except:
    print("오류")
"""
# 클래스의 정의?

class Dog: 
    def __init__(self, name): # __init__ : 클래스 객체를 생성할 때 일어나는 일을 정의하는 메소드. 클래스 초기화에 필요한 내용을 담는다.(생성자? 인 셈이 아닐까)
        self.name = name # self를 this로 이해하면 될듯. this.name = name 인 셈.
    def respond_to_command(self, command):
        if command == self.name: self.speak()
    def speak(self):
        print("멍멍")
        
fido = Dog("멍구") # 멍구 라는 이름의 Dog 객체 생성.
fido.respond_to_command("야옹이") # Dog 객체 중에 야옹이 가 없으므로 아무일도 일어나지 않는다.
fido.respond_to_command("멍구") # Dog 객체 중에 멍구 가 있으므로 speak함수가 호출된다.



















  