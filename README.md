# LEVEL2 U Stage - Deep Knowledge Tracing

<img src="https://img.shields.io/badge/Ubuntu-E95420?style=flat-square&logo=Ubuntu&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"> <img src="https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=Numpy&logoColor=white">

&nbsp;
### 2️⃣ RecSys_2조 2️⃣
![image](img/%E1%84%87%E1%85%A2%E1%84%82%E1%85%A5.png)
&nbsp;

###  최종 결과


&nbsp;

## 🙋🏻‍♂️🙋🏻‍♀️  Member
<table align="center">
  <tr height="155px">
    <td align="center" width="150px">
      <a href="https://github.com/ktasha45"><img src="https://avatars.githubusercontent.com/ktasha45"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/NIckmin96"><img src="https://avatars.githubusercontent.com/NIckmin96"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/parkkyungjun"><img src="https://avatars.githubusercontent.com/parkkyungjun"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/HeeJeongOh"><img src="https://avatars.githubusercontent.com/HeeJeongOh"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/yhw991228"><img src="https://avatars.githubusercontent.com/yhw991228"/></a>
    </td>
  </tr>
  <tr height="80px">
    <td align="center" width="150px">
      <a href="https://github.com/ktasha45">김동영_4028</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/NIckmin96">민복기_T4074</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/parkkyungjun">박경준_T4076</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/HeeJeongOh">오희정_T4129</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/yhw991228">용희원_T4130</a>
    </td>
  </tr>
</table>
&nbsp;

## 🖼️ Project Overview

초등학교, 중학교, 고등학교, 대학교와 같은 교육기관에서 우리는 시험을 늘 봐왔습니다. 시험 성적이 높은 과목은 우리가 잘 아는 것을 나타내고 시험 성적이 낮은 과목은 반대로 공부가 더욱 필요함을 나타냅니다. 시험은 우리가 얼마만큼 아는지 평가하는 한 방법입니다.

하지만 시험에는 한계가 있습니다. 우리가 수학 시험에서 점수를 80점 받았다면 우리는 80점을 받은 학생일 뿐입니다. 우리가 돈을 들여 과외를 받지 않는 이상 우리는 우리 개개인에 맞춤화된 피드백을 받기가 어렵고 따라서 무엇을 해야 성적을 올릴 수 있을지 판단하기 어렵습니다. 이럴 때 사용할 수 있는 것이 DKT입니다!

DKT는 Deep Knowledge Tracing의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론입니다.

시험을 보는 것은 동일하지만 단순히 우리가 수학을 80점을 맞았다고 알려주는 것을 넘어서 우리가 수학이라는 과목을 얼마만큼 이해하고 있는지를 측정하여 줍니다. 게다가 이런 이해도를 활용하여 우리가 아직 풀지 않은 미래의 문제에 대해서 우리가 맞을지 틀릴지 예측이 가능합니다!

이런 DKT를 활용하면 우리는 학생 개개인에게 수학의 이해도와 취약한 부분을 극복하기 위해 어떤 문제들을 풀면 좋을지 추천이 가능합니다! 그렇기 때문에 DKT는 교육 AI의 추천이라고도 불립니다. DKT는 맞춤화된 교육을 제공하기 위해 아주 중요한 역할을 맡게 됩니다.

우리는 대회에서 Iscream 데이터셋을 이용하여 DKT모델을 구축할 것입니다. 다만 대회에서는 학생 개개인의 이해도를 가리키는 지식 상태를 예측하는 일보다는, 주어진 문제를 맞출지 틀릴지 예측하는 것에 집중할 것입니다! 우리는 각 학생이 푼 문제 리스트와 정답 여부가 담긴 데이터를 받아 최종 문제를 맞출지 틀릴지 예측할 것입니다!


&nbsp;

## 📀 Dataset
&nbsp; 
### INPUT
&nbsp; 

- ```train_data.csv``` : 
&nbsp; 



## 🗂️ Folder Structure

  ```
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
