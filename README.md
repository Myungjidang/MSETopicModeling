# Multilingual Semantic Embedding Space Topic Modeling

<p align = "center"><img src = https://github.com/Myungjidang/MSETopicModeling/assets/133189623/bbac75d2-fd0e-47f3-9e91-bf2d24f2e891 width=70%></p>

![model](https://github.com/Myungjidang/MSETopicModeling/assets/133189623/ab907426-0274-4ab7-9a5a-94dbb4564c7e)

<br>
해당 모델은 '직교성 제약을 이용한 단일언어 의미보존 다국어 토픽모델링'에 대한 코드이다.



## Dataset

사용된 데이터는 실제로 Google Patent에서 확보한 13만여건의 5개 국어(한글, 영어, 중국어, 일본어, 독일어) 특허 데이터이다:

- **Google 특허 데이터**: 디스플레이 기술 관련 5개 국어 특허 데이터로 각 언어 데이터마다 특허의 여러 항목 중 발명의 명칭, 대표청구항, 요약, 출원일 등 15개의 Field로 구성되어 있다.

- ***한국어 데이터*** : 30,318개
  
- ***영어 데이터*** : 49,330개
  
- ***중국어 데이터*** : 39,057개
  
- ***일본어 데이터*** : 10,738개
  
- ***독일어 데이터*** : 9,403개

- ***데이터 링크*** : <https://drive.google.com/drive/folders/170gwdYhpbtMbtjbeCO4EjVg6K-tHx4cf?usp=drive_link>



## Output
**토픽 모델링 결과물**
- ***Image*** : output / Topic_modeling / Treemap : 각 토픽 별 키워드 추출에 관한 html 파일

- ***Excel*** : output / Topic_modeling / Topic_check_multi : 토픽 분석에 관한 xlsx 파일



**시계열 분석 결과물**
- ***Images*** :

output / images / scala / total : 특허 주제가 갖는 실제 데이터 양과 이전 연도의 평균/VAR 알고리즘을 사용한 데이터 양 예측 비교

output / images / vector / var : 실제 특허 중심의 이동과 VAR 알고리즘을 활용해 추적한 중심 (실행 파일에 출력된 Result of timeseries forecasting 참고)

- ***Wordlist*** :
  
output / wordlist / total : 실제 특허 주제의 중심에서 가장 가까운 단어

output / wordlist / var : var로 예측한 특허 주제의 중심에서 가장 가까운 단어

output / wordlist / country_total : 실제 특허 주제의 중심에서 가장 가까운 단어(국가별)

output / wordlist / country_var : var로 예측한 특허 주제의 중심에서 가장 가까운 단어(국가별)

- ***Docxlist*** :
  
output / docxlist / total : 실제 특허 주제의 중심에서 가장 가까운 document

output / docxlist / var : var로 예측한 특허 주제의 중심에서 가장 가까운 document
