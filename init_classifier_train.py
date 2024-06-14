from torchvision import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import os
import warnings 
warnings.filterwarnings("ignore")

'''

용도: 9종 분류를 위한 Resnet 학습

실행 방법 
vehicle_classifier 폴더로 이동 후 터미널에
python Init_classifier_train.py 입력

코드실행 결과
초기 차량 분류기(9종에 대해서 분류)인 resnet-50 학습 및 가중치 저장
(저장경로: ./res_weight/resnet50/final)

'''

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
        label -=1
        return image,label
        
import albumentations as A

train_data = A.Compose([
    A.Resize(224, 224),

    # A.OneOf([
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomRotate90(p=0.5),
    #     A.VerticalFlip(p=0.5)
    # ], p=0.5),

    A.OneOf([
        A.MotionBlur(p=0.5),
        A.OpticalDistortion(p=0.5),
        A.GaussNoise(p=0.5)
    ], p=0.5),

    A.OneOf([
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(p=0.5),
        A.RandomBrightnessContrast(p=0.5)
    ], p=0.5),

    A.pytorch.ToTensorV2()
])

test_data = A.Compose([                                
    A.Resize(224, 224),
    A.pytorch.ToTensorV2()
    
])

# gpu = int(input("0 or 1: "))       
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class ResNetModel(nn.Module):
    def __init__(self,name='resnet50', num_classes=5, train_only_last_layer=False, pretrained=True):
        super(ResNetModel, self).__init__()
        resnets={'resnet18':models.resnet18(pretrained=pretrained),'resnet50':models.resnet50(pretrained=pretrained),'resnet101':models.resnet101(pretrained=pretrained),'resnet152':models.resnet152(pretrained=pretrained)}
        self.resnet = resnets[name]
        if train_only_last_layer:
            for param in self.resnet.parameters():
                param.requires_grad = False
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    

def train(model, dataloader, optimizer, criterion, epoches, device, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    best_f1_score = 0.0
    best_weights = None
    best_epoch = None
    # best_valid_loss = float('inf')
    # patience = 5
    # current_patience = 0

    for epoch in range(1, epoches + 1):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(dataloader):
            images = images.float().to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            train_pred = []
            train_labels = []
            valid_loss = 0.0
            valid_preds = []
            valid_labels = []
            
            print('Training performance validation')
            for images, labels in tqdm(train_valid_loader):
                images = images.float().to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()
                for pred,label in zip(preds,labels):
                    train_pred.append(pred)
                    train_labels.append(label)
           
     
            train_f1 = f1_score(train_labels,  train_pred,average='macro')
            train_acc = accuracy_score(train_labels,  train_pred)

            print(f'Epoch {epoch}, Train_Loss: {train_loss/len(dataloader):.4f},  Train_F1: {train_f1:.4f}, Train_Accuracy: {train_acc:.4f}')
            classes_acc(train_labels, train_pred,mode)

            print('Validation performance validation')
            for images, labels in tqdm(valid_loader):
                images = images.float().to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()
                for pred,label in zip(preds,labels):
                    valid_preds.append(pred)
                    valid_labels.append(label)

            valid_f1 = f1_score(valid_labels, valid_preds,average='macro')
            valid_acc = accuracy_score(valid_labels, valid_preds)
            valid_loss=valid_loss/len(valid_loader)
            print(f'Epoch {epoch}, Valid_Loss: {valid_loss:.4f}, '
                  f' Valid_F1: {valid_f1:.4f}, Valid_Accuracy: {valid_acc:.4f}')
            classes_acc(valid_labels, valid_preds,mode)

            # 최고 성능 모델 저장
            if valid_f1 > best_f1_score:
                best_f1_score = valid_f1
                best_weights = model.state_dict()
                best_epoch = epoch
   
            
            print(f'best_epoch: {best_epoch}')
            print(f'best_f1_score: {best_f1_score:.4f}')
            
            # #조기 종료 체크
            # if valid_loss < best_valid_loss :
            #     best_valid_loss = valid_loss
            #     current_patience = 0
            # else:
            #     current_patience += 1
            #     if epoch > 10 and current_patience >= patience:
            #         print(f"Early stopping at epoch {epoch}")
            #         break

            # 모델 저장
            torch.save({'model_state_dict': best_weights,
                        'epoch': best_epoch
                    }, f'{save_path}best_weights.pth')

                    


def classes_acc(labels,preds):
    classNames = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 'Semi','FUll']
 
    classes = [0] * len(classNames)
    correct = [0] * len(classNames)
    res=[]
    for label, pred in zip(labels, preds):
        classes[label] += 1

        if label == pred:
            correct[label] += 1

    for i in range(len(classNames)):
        accuracy = correct[i] / classes[i] if classes[i] != 0 else 0.0
        res.append(f"Accuracy of class {classNames[i]}: {accuracy:.4f}")
        print(res[i])



mode = '9_classes'
   
train_csv = f'./data/{mode}_train.csv'
valid_csv = f'./data/{mode}_valid.csv'
test_csv = f'./data/{mode}_test.csv'

trainset = vehicleDataset(train_csv, transform = train_data)
train_validset = vehicleDataset(train_csv,  transform = test_data)
validset = vehicleDataset(valid_csv,  transform = test_data)
testset = vehicleDataset(test_csv,  transform = test_data,infer = True)


train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=4)

train_valid_loader = torch.utils.data.DataLoader(train_validset, batch_size=32,
                                          shuffle=False, num_workers=4)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=32,
                                         shuffle=False, num_workers=4)

test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=4)


'''
용도: 9종 분류를 위한 Resnet 학습

실행 방법 
vehicle_classifier 폴더로 이동 후 터미널에
python Init_classifier_train.py 입력

코드실행 결과
초기 차량 분류기(9종에 대해서 분류)인 resnet-50 학습 및 가중치 저장
(저장경로: ./res_weight/resnet50/final)
'''

if __name__ == '__main__': 
   
    num_classes = int(mode.split('_')[0])
    epoches = 100
    criterion = torch.nn.CrossEntropyLoss()
    modelname = 'resnet50' # resnet18, resnet50,reset101,renet152 중 하나 선택
    save_mid_path = f'./res_weight/{modelname}/finetune/'
    load_model_path = save_mid_path+'best_weights.pth'
    save_final_path = f'./res_weight/{modelname}/final/'
    
    finetune_model = ResNetModel(modelname,num_classes=num_classes, train_only_last_layer=True, pretrained=True)
    finetune_model = finetune_model.to(device)
    optimizer = torch.optim.Adam(finetune_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    print("start training finetune_model training")
    train(finetune_model,train_loader,optimizer,criterion,epoches,device,save_mid_path)
    print("training finetune_model finished")

    print("final model")
    final_model=ResNetModel(modelname,num_classes=num_classes, train_only_last_layer=False, pretrained=False)
    best_weights = torch.load(load_model_path)
    final_model.load_state_dict(best_weights['model_state_dict'])
    final_model = final_model.to(device)
    optimizer= torch.optim.Adam(final_model.parameters(), lr=0.0001)

    print("start training final model")
    train(final_model,train_loader,optimizer,criterion,epoches,device,save_final_path)
    print("training final model finished")