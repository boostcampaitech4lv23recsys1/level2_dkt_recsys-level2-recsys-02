# Deep Knowledge Tracing
![image](https://user-images.githubusercontent.com/47550287/206654876-7cecf8f7-7233-455c-83df-e26283c68632.png)

## 팀원 소개
|이름|역할|
|----|---|
|[박경준](https://github.com/parkkyungjun)|last query, boosting, gcn, tuning을 시도|
|[민복기](https://github.com/NIckmin96)|EDA, FE 및 1D-CNN 모델 실험|
|[김동영](https://github.com/ktasha45)|lightGCN + sequence model, lightGCN 개량, 튜닝 시도|
|[오희정](https://github.com/HeeJeongOh)|EDA, FE를 수행하고 베이스라인 제공 모델 외에도 다양한 ML, DL모델들을 실험|
|[용희원](https://github.com/yhw991228)| EDA, FE 및 딥러닝 모델 위주 실험|

## 프로젝트 주제
**Iscream 데이터셋을 활용해 DKT모델을 구축해 주어진 문제의 정답 여부를 예측하는 이진분류 Task**

DKT는 Deep Knowledge Tracing의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론이다. 이런 DKT를 활용하면 우리는 학생 개개인에게 수학의 이해도와 취약한 부분을 극복하기 위해 어떤 문제들을 풀면 좋을지 추천이 가능하기 때문에 DKT는 교육 AI의 추천이라고도 불린다. 즉, DKT는 맞춤화된 교육을 제공하기 위해 아주 중요한 역할을 맡게 된다.  
 
## 프로젝트 개요
Iscream 데이터셋을 이용하여 DKT 모델을 구축하고, 학생 개개인의 이해도를 가리키는 지식 상태를 예측하는 일보다는 주어진 문제를 맞출지 틀릴지 예측하는 것에 집중한다. 각 학생이 푼 문제 리스트와 정답 여부가 담긴 데이터를 받아 최종 문제를 맞출지 틀릴지 예측한다.
- input : 약 7000명의 사용자들의 문제 풀이 내역
- output : test_data 사용자들의 마지막 문제의 정답여부
- metrics : **AUROC(Area Under the ROC curve), Accuracy**

## 데이터셋 구조
![image](https://user-images.githubusercontent.com/62127798/207546076-3f496b7b-1452-42b2-a9ba-23edddb3116b.png) 

| 이름 | 설명 |
| --- | --- |
| **userID** | 사용자 고유 번호  |
| **assessmentItemID** | 문항의 고유번호 |
| **testID** |  시험지의 고유번호 |
| **answerCode** | 사용자가 해당 문항을 맞췄는지 여부 (binary) |
| **Timestamp** | 사용자가 해당문항을 풀기 시작한 시점 |
| **KnowledgeTag** | 문항 당 하나씩 배정되는 태그로, 일종의 중분류 역할 |

## 베이스라인 구조
```bash
├── README.md
├── dkt                      # sequential 접근
│   ├── src
│   │   ├── criterion.py     # loss 함수 : BCELoss
│   │   ├── dataloader.py    # 전처리 및 학습가능한 input생성
│   │   ├── metric.py        # AUROC, ACC
│   │   ├── model.py         # LSTM, LSTM+ATTN, BERT
│   │   ├── optimizer.py     # Adam, AdamW
│   │   ├── scheduler.py     # learning rate 조절
│   │   ├── trainer.py       # train, validation
│   │   └── utils.py         # random seed 고정
│   ├── args.py              # 파라미터 설정
│   ├── inference.py         # 예측된 csv파일 생성
│   ├── requirements.txt
│   ├── train.py             # main
│   └── README.md
|
└── lightgcn                 # graph 접근
├── lightgcn
│   ├── datasets.py          # 전처리 및 학습가능한 input 생성
│   ├── models.py            # LightGCN
│   └── utils.py             # seed 고정, logger 설정
├── config.py                # 파라미터 설정
├── inference.py             # 예측된 csv파일 생셩
├── install.sh
├── train.py
└── README.md
```

## 프로젝트 수행결과
|||
|---|---|
|**데이터**| |
|**모델**| Catboost(0.425) + lightGBM(0.425) + lightGCN(0.15)
|**하이퍼파라미터**||
|**앙상블**||


## 최종 결과
public
![image](https://user-images.githubusercontent.com/62127798/207544836-bf441237-8b2d-449d-a21d-37a469859f89.png)
private
![image](https://user-images.githubusercontent.com/62127798/207544928-c4b420ac-3973-4234-9615-32501caeba5f.png)
