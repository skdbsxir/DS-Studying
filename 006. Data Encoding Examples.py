# Json File. {"Key" : "Value"} 형태.
# Key == 문자열, Value == JSON 객체 로 생각할 수 있음.
# Value로 오는 문자열은 " " 로 감싸고, 숫자나 boolean값(소문자)은 감싸지 않음.
import json
json_str = """{"name" : "Kim", "height" : 178}"""
my_obj = json.loads(json_str) # loads() : JSON을 Python Dictionary로 변환.
print(my_obj)
print(type(my_obj)) # 실제 타입이 dict로 나옴.
my_str = json.dumps(my_obj) # dumps() : Python Dictionary를 JSON으로 변환.
print(my_str)
print(type(my_str)) # 실제 타입이 str로 나옴.
print('\n')


# XML File. <tag (attribute)> value </tag> 형태.
# 엄격한 계층적 구조.
# XML은 내용에 순서가 있고, 파일에 쓰인대로 순서가 정해짐. JSON은 순서가 없는 트리구조.
# 간단한 데이터는 JSON, 복잡한 문서는 XML을 주로 사용.
# 사용하기 간단한 ElementTree를 XML을 파싱함. 보통은 DOM(Document Object Model)을 사용해서 불러옴. 웹 브라우저도 마찬가지로 DOM 사용.
import xml.etree.ElementTree as ET
xml_str = """
<data>
    <country name="Liechtenstein">
        <rank>1</rank>
        <year>2008</year>
        <gdppc>141100</gdppc>
        <neighbor name="Austria" direction="E" />
        <neighbor name="Switzerland" direction="W" />
    </country>
    <country name="Singapore">
        <rank>4</rank>
        <year>2011</year>
        <gdppc>59900</gdppc>
        <neighbor name="Malaysia" direction="N" />
    </country>
</data>
"""
root = ET.fromstring(xml_str) # xml 문자열 파싱.
print(type(root)) # xml.etree.ElementTree.Element
print(root.tag) # 이 xml 데이터의 최상위 태그 출력. 최상위 태그는 <data>이므로 data가 출력.
print(root[0]) # 0번째 자식 노드의 성분 (0번째는 country name="Liechtenstein")
print(root[1].attrib) # 1번째 노드의 세부정보 (1번째는 country name="Singapore")
print(root.getchildren()) # 해당 노드의 자식 노드 반환 (모든 자식노드 country를 반환)
modified_xml_str = ET.tostring(root) # 편집이 끝난 xml 객체를 문자열로 변환.
print('파싱된 타입 : ', type(modified_xml_str)) # bytes 클래스.
print('\n')


# HTML File. XML의 하위개념이자 웹에서 사용하는 언어.
# 일반적인 XML 문서보다 훨씬 복잡. 문서를 처리해 데이터를 추출하는 과정이 좀 까다로움. 문서 전체를 보고 필요한 정보를 찾아야 함.
# HTMLParser 클래스로 HTML문서를 처리. 문서를 처음부터 끝까지 읽어 태그나 문자열에 따라 내용을 처리함.
# 예제 코드 - 위키피디아 페이지를 불러오고, paragraph태그에 포함된 하이퍼링크 갯수를 count.
from html.parser import HTMLParser
import urllib.request

TOPIC = "Beatmania" # 검색 토픽 설정
url = "https://en.wikipedia.org/wiki/%s" % TOPIC # url 생성

class LinkCountingParser(HTMLParser) :
    def __init__(self) :
        # HTMLParser를 상속한 클래스를 정의.
        self.in_paragraph = False
        self.link_count = 0
        HTMLParser.__init__(self)
        
    def handle_starttag(self, tag, attrs) :
        # 시작 태그를 처리하는 방법을 정의.
        if tag == 'p':
            self.in_paragraph = True
        elif tag == 'a' and self.in_paragraph :
            self.link_count += 1
            
    def handle_endtag(self, tag) :
        # 종료 태그를 처리하는 방법을 정의.
        if tag == 'p' :
            self.in_paragraph = False
            
html = urllib.request.urlopen(url).read() # url에 해당하는 html 문서를 읽어옴.
html = html.decode('utf-8') # bytes를 문자열로 변환
parser = LinkCountingParser() # 클래스 인스턴스 생성 후
parser.feed(html) # 읽어온 html을 입력으로 넣는다.
print('문서엔 %d 개의 링크가 존재.' % parser.link_count)
print('\n')


# Tar 묶음 파일. 
# 폴더의 모든 파일과 하위 폴더를 하나의 파일로 묶음. 전송이나 보관에 용이.
# 데이터를 압축하지 않음. 그래서 보통 Tar파일을 다시 Gzip 같은 프로그램으로 압축하고 .tgz 또는 .tar.gz 확장자를 붙임.
# 일반적으로 명령행에서 묶음을 바로 해제함. 빅데이터 프로세싱 할때 압축해제 했던거 생각하면 됨.
"""
tar -xvf my_directory.tar # my_directory.tar 파일을 현재 위치에 묶음 해제.
tar -zxf file.tar.gz # 압축과 tar 묶음을 해제.
tar -cf Example.tar Homework # Example.tar 파일을 Homework 폴더에 묶음 해제.
"""
# Gzip 파일. 유닉스 계열 OS에서 많이 사용.
# 압축 속도가 느림. 하지만 압축률이 좋고, 압축해제가 빠르고, 파일 내 일부만 선택적으로 해제가 가능함.
# 디플레이트(DEFLATE)알고리즘 이용. 압축 후 파일이 블록으로 쪼개져 저장되고, 각 블록은 헤더와 데이터로 나뉨.
## 헤더에 있는 정보를 이용하면 해당 블록의 데이터를 1바이트 단위로 읽을 수 있음. 그래서 메모리를 효율적으로 사용할 수 있다.
"""
gzip myfile.txt # myfile.txt를 압축.
gunzip myfile.txt.gz # myfile.txt.gz를 압축 해제.
""" 
# Zip 파일. Gzip과 거의 동일하지만, Zip은 폴더를 압축할 수 있음.
"""
zip filename.zip input.txt input2.txt resume.doc pic1.jpg # 여러 파일을 filename.zip으로 압축.
unzip filename.zip # 현재 위치에 압축 해제.
"""


# 이미지 파일. 비트맵과 벡터 이미지로 나뉨.
## 비트맵 이미지는 2차원 배열에 각 픽셀의 값을 저장. 우리가 사용하는 이미지 대부분이 비트맵 이미지.
### 메타데이터, 픽셀 배열 값으로 구성. 옵션으로 데이터가 압축되어있을 수 있음. 
### 손실/무손실 형식, 흑백/컬러 형식. 흑백은 픽셀마다 값을 하나만 저장하지만 컬러는 픽셀마다 3가지 값(RGB)을 저장.
## 벡터 이미지는 직선이나 곡선을 수학적으로 나타내고 정보를 저장하는 형식. (ex. 직선의 시작점과 끝점을 저장) 확대/축소 해도 그림이 안깨짐. SVG 확장자.
"""
# `imread` is deprecated! `imread` is deprecated in SciPy 1.0.0. 얘네가 추천하지 않는다. plt.imread를 사용하는게 낫다고 함. (Use ``matplotlib.pyplot.imread`` instead.)
from scipy.ndimage import imread
img = imread('C:\\Users\\skdbs\\Desktop\\murmansk.jpg')
print(img)
"""
import matplotlib.pyplot as plt
img2 = plt.imread('C:\\Users\\skdbs\\Desktop\\murmansk.jpg')
print(img2) # 3차원 배열 반환. 처음 두 차원은 너비와 높이, 세번째 차원은 RGB채널 값.