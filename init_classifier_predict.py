from torchvision import models
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from tqdm import tqdm
import shutil
import albumentations as A
from sklearn.metrics import accuracy_score, f1_score
import warnings 
warnings.filterwarnings("ignore")

class vehicleDataset(Dataset):
    def __init__(self,csv_file,transform=None,infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img_path = self.data.iloc[idx,0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer: 
            image = self.transform(image=image)['image']
            return image 
        
        if self.transform:
            image = self.transform(image=image)['image']

        label = self.data.iloc[idx,1]
        label -= 1
        return image,label
        

test = A.Compose([
                                                
    A.Resize(224, 224),
    A.pytorch.ToTensorV2()
    
])


class ResNetModel(nn.Module):
    def __init__(self,name='resnet50', num_classes=5, train_last_layer=False, pretrained=True):
        super(ResNetModel, self).__init__()
        resnets={'resnet18':models.resnet18(pretrained=pretrained),'resnet50':models.resnet50(pretrained=pretrained),'resnet101':models.resnet101(pretrained=pretrained),'resnet152':models.resnet152(pretrained=pretrained)}
        self.resnet = resnets[name]
        if train_last_layer:
            for param in self.resnet.parameters():
                param.requires_grad = False
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def classes_acc(labels,preds):
   
    classNames = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 'Semi','Full']
   
        
    classes = [0] * len(classNames)
    correct = [0] * len(classNames)
    res=[]
    for label, pred in zip(labels, preds):
        classes[label] += 1

        if label == pred:
            correct[label] += 1

    for i in range(len(classNames)):
        accuracy = correct[i] / classes[i] if classes[i] != 0 else 0.0
        accuracy *= 100
        res.append(f"Accuracy of class {classNames[i]}: {accuracy:.2f}")
        print(res[i])
    
 

def predict(model, dataloader, device, weight_path,gtpath,pred_type=None,copy=False):
    best = torch.load(weight_path)
    print('best Epoch :', best['epoch'])
    model.load_state_dict(best['model_state_dict'])
    model = model.to(device)
    model.eval()
    res = []
    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.float().to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            for pred in preds:
                res.append(pred)
    
    gt_df = pd.read_csv(gtpath)
    imgPath = gt_df.iloc[:,0].values.tolist()   
    gtlist = gt_df.iloc[:,1].values.tolist()
    for i in range(len(gtlist)):
        gtlist[i]-=1

   # 9종에 대한 정확도, F1 Score 계산
    accuracy = accuracy_score(gtlist, res)
    accuracy *= 100
    f1 = f1_score(gtlist, res, average='macro')
    f1*=100
    classes_acc(gtlist, res,mode)
    type = gtpath.split('/')[-1]
    print(f'Final model {type} accuracy: {accuracy:.2f}')
    print(f'Final model {type} F1 Score: {f1:.2f}')


    csv_file = f'{pred_type}_pred.csv'
    for i in range(len(res)):
        res[i]+=1
        gtlist[i]+=1
    

    class12_csv = gtpath.replace('9_classes','12_classes')
    class12_df = pd.read_csv(class12_csv)
    class12_gt = class12_df .iloc[:,1].values.tolist()

    imgs = []
    for img in imgPath:
        imgs.append(img.split('/')[-1])
        


    result_df = pd.DataFrame({
        'img' : imgs,
        'label'  : gtlist, # 9종 
        'pred':   res, #9종
        'class12': class12_gt #12종
    })
        
    pred_path = './init_classification'
    os.makedirs(f'{pred_path}/csv',exist_ok=True)
    result_df .to_csv(f'{pred_path}/csv/{csv_file}' ,index=False)
    print('save result done')

    # 예측된 클래스에 맞게 폴더 생성 후 이미지 복사
    # if copy:
    #     for row in result_df.itertuples():
    #         src = row.img
    #         img = src.split('/')[-1]
    #         pred = row.pred
    #         folder = f'0{pred}'
            
    #         if pred == 8:
    #             folder = 'Semi'
    #         elif pred == 9:
    #             folder = 'Full'
            
                
    #         save_path =f'{pred_path}/{pred_type}/{folder}'
            
    #         os.makedirs(save_path,exist_ok=True)
    #         dst = f'{save_path}/{img}'
    #         shutil.copy(src,dst)
        
        

    #     print('copy done')
  


mode = '9_classes'
train_csv = f'./data/{mode}_train.csv'
valid_csv = f'./data/{mode}_valid.csv'
test_csv = f'./data/{mode}_test.csv'
gt_csv = f'./data/{mode}_gt.csv'


trainset = vehicleDataset(train_csv, transform=test,infer=True)
validset = vehicleDataset(valid_csv,  transform=test,infer=True)
testset  = vehicleDataset(test_csv,  transform=test,infer=True)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=False, num_workers=4)


valid_loader = torch.utils.data.DataLoader(validset, batch_size=32,
                                         shuffle=False, num_workers=4)

test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=4)
       
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




'''


용도: 9종에 대한 분류 성능 측정 및 예측 결과 저장

실행 방법
vehicle_classifier로 이동 후 터미널에 
Python Init_classifier_predict 입력

코드실행 결과
./res_weight/resnet50/final에 저장된 모델 가중치를 불러와서
train, valid, test 데이터에 대한 차종 분류 수행  
각 데이터 셋에 대한 정확률 출력

init_classification 폴더에 결과값을 담은 csv 파일 생성 
(저장경로: ./init_classification/csv)

데이터 타입별 폴더 생성 이후 해당 폴더 하위에 클래스 별 폴더 생성  
이후 클래스별 폴더에 해당 클래스로 예측한 이미지 복사
(저장경로: ./init_classification/)



'''

if __name__ == '__main__': 
    modelnames=['resnet18','resnet50','resnet101','resnet152']
    modelname ='resnet50'

    num_classes = int(mode.split('_')[0])
    model=ResNetModel(modelname, num_classes=num_classes, train_last_layer=False, pretrained=False)
    # mid_weight_path = f'./res_weight/{modelname}/mid/best_weights.pth'
    # print('middle')
    # print('train predict')
    # predict(model, train_loader, device, mid_weight_path,train_csv,'train')
    # print('valid predict')
    # predict(model,  valid_loader, device, mid_weight_path,valid_csv,'valid')
    # print('test predict')
    # predict(model, test_loader, device, mid_weight_path,gt_csv,'test')
   
    final_weight_path = f'./res_weight/{modelname}/final/best_weights.pth'
    print('final')
    print('train predict')
    predict(model, train_loader, device, final_weight_path,train_csv,'train')
    print('valid predict')
    predict(model,  valid_loader, device, final_weight_path,valid_csv,'valid')
    print('test predict')
    predict(model, test_loader, device, final_weight_path,gt_csv,'test')