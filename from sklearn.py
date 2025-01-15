from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("슝~")

# 데이터 불러오기
bc = load_breast_cancer()
print(type(dir(bc)))  

#bc에 어떤 정보들이 담겨있는지 확인하기
print("bc에 담겨진 정보:", bc.keys())

#Feature Data 지정하기
bc_data = bc.data

#bc_data 크기 확인해보기(배열의 형상정보 출력하기)
print(bc_data.shape)

#샘플로 bc_data에서 하나의 데이터만 확인해 보기)
print( bc_data[0] )

bc_label = bc.target
print(bc_label.shape)
print(bc_label)

#라벨의 이름을 출력해 봅시다. 
print(bc.target_names)
# benign : (of a disease) not harmful in effect.
# malignant : (of a disease) very virulent or infectious.

#데이터의 설명이 담겨있는 변수를 출력해 봅시다. 
print(bc.DESCR)

#feature에 대한 설명이 담긴 변수를 출력해 봅시다. 
print(bc.feature_names)

# 데이터셋 파일이 저장된 경로를 출력해 봅시다. 
print(bc.filename)

#pd라는 약어로 판다스 사용하기
import pandas as pd 


#판다스 버전 출력하기
print(pd.__version__)

#유방암 데이터셋을 pandas가 제공하는 DataFrame이라는 자료형으로 변환하기 
bc_df = pd.DataFrame(data=bc_data, columns=bc.feature_names)
bc_df

# sklearn.model_selection 패키지의 train_test_split을 활용
from sklearn.model_selection import train_test_split

# trainig dataset과 test dataset을 간단히 분리해 봅시다.
X_train, X_test, y_train, y_test = train_test_split(bc_data, bc_label, test_size = 0.2, random_state=7)

print('X_train 개수: ', len(X_train),', X_test 개수: ', len(X_test))
print('y_train 개수: ', len(y_train),', y_test 개수: ', len(y_test))

#사이킷런의 의사결정트리를 import 합니다. 
from sklearn.tree import DecisionTreeClassifier

#의사결정 트리를 선언합니다. 
decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)

#훈련데이터로 의사결정트리를 학습합니다. 
decision_tree.fit(X_train, y_train)

#사이킷런의 랜덤포레스트를 import 합니다. 
from sklearn.ensemble import RandomForestClassifier

#랜덤 포레스트를 생성합니다. 
random_forest = RandomForestClassifier(random_state=32)
print(random_forest._estimator_type)

#훈련데이터로 랜덤포레스트를 학습합니다. 
random_forest.fit(X_train, y_train)

#사이킷런의 svm을 import 합니다. 
from sklearn import svm

#svm을 생성합니다.
svm_model = svm.SVC()
print(svm_model._estimator_type)

#훈련데이터로 svm분류모델을 학습합니다. 
svm_model.fit(X_train, y_train)

#사이킷런의 SGD Classifier 모델을 import 합니다. 
from sklearn.linear_model import SGDClassifier

#sgd classifier를 생성합니다. 
sgd_model = SGDClassifier()
print(sgd_model._estimator_type)

#훈련데이터로 SGC분류기를 학습합니다. 
sgd_model.fit(X_train, y_train)

#사이킷런의 로지스틱 회귀를 import 합니다. 
from sklearn.linear_model import LogisticRegression

#의사결정 트리를 선언합니다. 
logistic_model = LogisticRegression()
print(logistic_model._estimator_type)

#훈련데이터로 로지스틱 회귀를 학습합니다. 
logistic_model.fit(X_train, y_train)

#모델의 predict 함수를 이용해서 X_test를 예측해 주세요 
y_pred = decision_tree.predict(X_test)
print(y_pred)

# sklearn.metrics의 accuaracy_score함수를 import 해 주세요
from sklearn.metrics import accuracy_score

#y_test 와  y_pred를 인자로 넣어 정확도를 구해 주세요
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#직접 코드를 작성 해 보세요!
from sklearn.metrics import accuracy_score

model_list = [decision_tree ,random_forest, svm_model, sgd_model, logistic_model]
y_pred_list =[]

for model in model_list : 
    y_pred = model.predict(X_test)
    y_pred_list.append(y_pred)

acc_list=[]

for y_pred in y_pred_list :
    accuracy = accuracy_score(y_test,y_pred)
    acc_list.append(accuracy)

print(acc_list)