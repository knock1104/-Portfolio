# ML_competition
___
### Competition 소개
***
#### 2020 Kaggle 에서 진행된 머신러닝 백화점 구매 고객 성별 예측 competition 입니다.

#### 대회기간
2020.09 ~ 2020.12 (약 3개월)

#### 데이터 소개

* transaction.csv >> 롯데백화점 고객 구매데이터
* features.csv >> transaction data를 이용해 만들어진 features
* target.csv / y_train >> 여성 0 남 1로 분류된 성별 데이터
* train/test_transactions.csv transaction data를 학습을 위해 train과 test로 나눠놓은 데이터
* ML_feature1 >> 학습을 위해 만든 features 1
* ML_feature2 >> 학습을 위해 만든 features 2
* ML_model1 >> ML_feature를 학습시키고 앙상블을 통해 성별을 예측하는 코드

##### 데이터 설명

* 롯데백화점 구매데이터 
* cust_id : 고객 고유번호
* tran_date : 구매시간
* store_nm : 구매지점
* goods_id : 상품고유번호
* gds_grp_nm : 상품소분류
* gds_grp_mclas_nm : 상품대분류
* amount : 구입금액

#### 버젼 확인
* python 3.9.13
* pandas 1.5.0 이상 권장

