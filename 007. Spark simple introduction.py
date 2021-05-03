"""
Linux 환경에서의 실행 필요. 빅데이터 프로세싱에서 보고 했던 것이랑 똑같이.
bin/spark-submit --master yarn-client myfile.py # 클러스터를 이용해 실행. 독립 스크립트로 실행. 파일 안에 설정을 작성해야 함.
bin/spark-submit --master local myfile.py # 로컬모드로 실행.
bin/spark-submit --master yarn-client # 대화형 스크립트로 실행. 객체 생성할 필요 없이 대화형으로 치면 됨.
"""

# SparkContext 객체 생성.
from pyspark import SparkConf, SparkContext
conf = SparkConf()
sc = SparkContext(conf=conf)

# 파일의 각 행을 읽고, 클러스터로 분산시켜 Spark RDD 생성.
lines = open("C:\\Users\\skdbs\\Desktop\\007. Word Example.txt")
lines_rdd = sc.parallelize(lines)

# 문장 부호를 없애고 모든 행을 소문자로 변환.
def clean_line(s) :
    s2 = s.strip().lower()
    s3 = s2.replace(".", "").replace(",","")
    return s3

lines_clean = lines_rdd.map(clean_line)

# 각 행을 단어 단위로 쪼갬.
words_rdd = lines_clean.flatMap(lambda l: l.split()) # Map Reduce 과정 中 Map.

# 각 단어의 등장 횟수 count.
def merge_counts(count1, count2) :
    return count1 + count2

# key가 있는 RDD 생성. key == 각 단어, value == 단어가 등장한 횟수.
# RDD는 결과를 출력하는 시점이 되면 생성되고 작업을 통해 최종 출력이 일어남. --> 실제로 필요한 시점에서만 존재. (게으른 방식) (?)
words_w_1 = words_rdd.map(lambda w: (w, 1))
counts = words_w_1.reduceByKey(merge_counts) # Map Reduce 과정 中 Reduce.

# 횟수를 총 합해서 출력.
for word, count in counts.collect() :
    print("%s: %i " % (word, count))