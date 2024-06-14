import pandas as pd 
import os
from pycaret.classification import *
import warnings
warnings.filterwarnings("ignore")

def ml_train(df,path):
   
    train = df
    train = train.drop(columns=['img'])

    clf = setup(data = train, target = 'label',fold=5,index=False,use_gpu=False,session_id=777,verbose=True) # 예측 값 및 학습 데이터 설정
    clf.model = compare_models(sort='Accuracy', fold = 5, n_select=1,exclude=['lightgbm'],verbose=True) # 가장 성능이 좋은 모델을 선정
    final_model = finalize_model(clf.model)
    save_model(final_model,path)
    print()


'''
용도: 트레일러 분류기 학습 및 모델 선정

실행 방법 
vehicle_classifier 폴더로 이동 후 터미널에
trailer_classifier_train.py입력

코드실행 결과
앞의 과정에서 추출된 특징을 이용하여 학습 진행 

Full과 Semi 각각에서 가장 우수한 분류기를 선택 후 모델 저장
(저장경로: ./ml/model)

'''

def main():
    full_df = pd.read_csv('./ml/train/csv/Full_wheels.csv')
    semi_df = pd.read_csv('./ml/train/csv/Semi_wheels.csv')
    model_path = './ml/model/'
    
    ml_train(semi_df,os.path.join(model_path,'semi_model'))
    ml_train(full_df,os.path.join(model_path,'full_model'))

if __name__ == "__main__":
    main()