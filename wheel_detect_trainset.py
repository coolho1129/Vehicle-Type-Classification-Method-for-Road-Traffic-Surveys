import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lang_sam import LangSAM
import os
import natsort
import torchvision
import torch
import pandas as pd
warnings.filterwarnings("ignore")
from pycaret.classification import *
from tqdm import tqdm
'''

https://github.com/luca-medeiros/lang-segment-anything
https://github.com/IDEA-Research/GroundingDINO

용도: 트레일러 데이터 분류기 학습을 위한 특징 추출(바퀴 수 및 바퀴 위치)

실행 방법 
vehicle_classifier 폴더로 이동 후 터미널에
python wheel_detect_trainset.py입력

코드실행 결과
바퀴를 탐지한 이미지 저장 (저장경로: ./ml/train/csv)

바퀴 수 및 위치를 csv파일에 저장(정답  label로 함께 저장됨)
(저장경로: ./ml/train/Full or ./ml/train/Semi)


'''

def remove_duplicates(boxes,masks, phrases,scores,score_treshold, threshold = 0.075):
    final_boxes, final_masks, final_phrases, final_scores = [], [] ,[] ,[]
  
    # NMS를 적용합니다.
    if boxes:
        selected_indices = torchvision.ops.nms(torch.stack(boxes), torch.tensor(scores), threshold)

        # 선택된 바운딩 박스들을 반환합니다.
        new_boxes = [boxes[i] for i in selected_indices]
        new_scores = [scores[i] for i in selected_indices]
        new_masks = [masks[i] for i in selected_indices]
        new_phrases = [phrases[i] for i in selected_indices]
        avg_width,  avg_height = calc_avg(boxes)
        height_treshold = avg_height * 0.7
        width_treshold = avg_width * 0.35

        if len(new_boxes)<7:
            height_treshold =  avg_height * 0.6
            width_treshold = avg_width * 0.3


        for box, mask, phrases, logit in zip(new_boxes, new_masks, new_phrases, new_scores):
            x_min, y_min, x_max, y_max = box
            box_width = x_max - x_min
            box_height = y_max - y_min
            confidence_score = round(logit.item(), 2)

            if  box_height<height_treshold or box_width<width_treshold or confidence_score<score_treshold:
                    continue
            # print(f'wheels{idx+1}')
            # print(x_min,x_max,y_min,y_max)
            # print('높이: ',box_height,'너비: ',box_width, '넓이: ', box_height * box_width)
            
            final_boxes.append(box)
            final_masks.append(mask)
            final_scores.append(logit)
            final_phrases.append(phrases)
        

    return final_boxes, final_masks, final_phrases, final_scores

def calc_avg(boxes):
    total_width = 0
    total_height = 0
    box_sizes = []
    box_heights = []
    box_widthes = []

    # 박스의 넓이를 구하고 리스트에 추가
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        box_width = x_max - x_min
        box_height = y_max - y_min
        box_area = box_width * box_height
        
        box_heights.append(box_height)
        box_widthes.append(box_width)
     
        total_width += box_width
        total_height += box_height
       

    # 리스트에 추가된 넓이 중에서 최소값과 최대값을 찾음
    min_height = min(box_heights)
    max_height = max(box_heights)
    min_width = min(box_widthes)
    max_width = max(box_widthes)

    avg_width = (total_width - min_width - max_width) / (len(box_widthes) - 2)
    avg_height = (total_height - min_height - max_height) / (len(box_heights) - 2)
    
    return avg_width,  avg_height

def outlier_removal(boxes,masks, phrases, logits, score_treshold):
    new_boxes, new_masks, new_phrases, new_logits = remove_duplicates(list(boxes),masks, phrases, logits,score_treshold=score_treshold)
    return new_boxes, new_masks, new_phrases, new_logits

def save_image_with_boxes(image, boxes, logits, mode,types,name,save_path = None):
    fig, ax = plt.subplots()
    ax.imshow(image)
    # ax.set_title("Image with Bounding Boxes")
    ax.axis('off')

    for i,(box, logit) in enumerate(zip(boxes, logits)):
        confidence_score = round(logit.item(), 2) 
        x_min, y_min, x_max, y_max = box
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(x_min, y_min, f"wheels{i+1}", fontsize=8, color='red', verticalalignment='top')
        ax.text(x_max, y_max, f"{confidence_score}", fontsize=8, color='red',verticalalignment='bottom')
    
    if not save_path:
        save_path = f'./ml/train/{mode}/{types}/box/'
    os.makedirs(save_path, exist_ok=True)
    save_path = save_path + f'{name}_box.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show(block=True)


def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height
    

'''
https://github.com/luca-medeiros/lang-segment-anything
https://github.com/IDEA-Research/GroundingDINO

용도: 트레일러 데이터 분류기 학습을 위한 특징 추출(바퀴 수 및 바퀴 위치)

실행 방법 
vehicle_classifier 폴더로 이동 후 터미널에
python wheel_detect_trainset.py입력

코드실행 결과
바퀴를 탐지한 이미지 저장 (저장경로: ./ml/train/csv)

바퀴 수 및 위치를 csv파일에 저장(정답  label로 함께 저장됨)
(저장경로: ./ml/train/Full or ./ml/train/Semi)
'''
def main():
    model = LangSAM("vit_h")
    text_prompt = "wheel"
    score_treshold = 0.4 
   
    for types in ['Full', 'Semi']:
        
        path = f"./ml/rawdata/{types}/"
        names = natsort.natsorted(os.listdir(path))
        labels = []
        wheel_cnt = []
        center_points = []
        for name in tqdm(names):
            label = int(name.split('_')[0])
            img_path = path + name
            width, height = get_image_size(img_path)
            image_pil = Image.open(img_path).convert("RGB")
            masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
            new_boxes,new_masks,new_phrases,new_logits = outlier_removal(boxes,masks, phrases, logits,score_treshold)

            if len(masks) == 0:
                        print(f"No objects of the '{text_prompt}' prompt detected in the image.")
            else:
                #segmetation code라서 사용하지 않음
                # # Convert masks to numpy arrays
                # masks_nps = [mask.squeeze().cpu().numpy().astype('uint8') for mask in new_masks]

                # # Display the image with bounding boxes and confidence scores
                # save_image_with_boxes(image_pil, new_boxes, new_logits,types,name)
                # detect_mask_path = f'ml/train/{types}/masked/'
                # os.makedirs(detect_mask_path,exist_ok=True)
                # detect_mask_path = detect_mask_path+f'mask{name}'
                # overlay_mask_on_image(image_pil, new_masks, detect_mask_path)
                
                # # Save the masks
                # mask_path = f"ml/train/{types}/image_mask/{name}/"
                # os.makedirs(mask_path, exist_ok=True)
                # for i, mask_np in enumerate(masks_nps):
                #     mask_save_path = mask_path+f'mask{i+1}.png'
                #     save_mask(mask_np, mask_save_path)

           
                wheel_cnt.append(len(new_boxes))
                labels.append(label)
                cps = [(-1,-1) for _ in range(12)]
                for idx, new_box in enumerate(new_boxes):
                    x_min, y_min, x_max, y_max = map(int, new_box)
                    # 바운딩 박스 중점 계산
                    cx = (x_max+x_min)/2
                    cy = (y_max+y_min)/2

                    # 상대 좌표로 변환
                    cx = cx / width
                    cy = cy / height

                    
                    cps[idx] = (cx,cy)
                     
                    
                center_points.append(cps)

        print("detection done")
        
        if len(names)==len(wheel_cnt):

            res_df =  pd.DataFrame({
                        'img':names,
                        'wheel':wheel_cnt
                    })
        
            transposed = list(map(list, zip(*center_points)))

            for idx, row in enumerate(transposed):
                keyX = f'center_x{idx+1}'
                keyY = f'center_y{idx+1}'
                # print(row)
                center_x = [centerP[0] for centerP in row]
                center_y = [centerP[1] for centerP in row] 
                res_df[keyX] = center_x
                res_df[keyY] = center_y

            res_df['label'] = labels

            
            os.makedirs('./ml/train/csv/wheels',exist_ok=True)
            wheels_path = f'./ml/train/csv/{types}_wheels.csv'
            res_df.to_csv(wheels_path, index=False)
            print(f'save {wheels_path}')
        else:
             print('모든 이미지에 대해 작업이 완료되지 않았습니다.')

if __name__ == "__main__":
    main()

#segmetation code라서 사용하지 않음
# def overlay_mask_on_image(image, masks, detect_path):
#     overlaid_image = np.array(image)

#     for  mask_np in masks:
#         # 확장된 마스크를 RGB 이미지 크기에 맞게 조절
#         expanded_mask = np.expand_dims(mask_np, axis=2)
#         expanded_mask = np.repeat(expanded_mask, 3, axis=2)

#         # 이미지에 마스크를 덧씌우기
#         overlaid_image = np.where(expanded_mask > 0, [255, 255, 0], overlaid_image)  # 노란색 마스크를 사용 (R=255, G=255, B=0)

#     overlaid_image = Image.fromarray(overlaid_image.astype(np.uint8))
#     overlaid_image.save(detect_path)

# def save_masked_image(image, masks, detect_path):
#     num_masks = len(masks)

#     fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
#     axes[0].imshow(image)
#     axes[0].set_title("Original Image")
#     axes[0].axis('off')

#     for i, mask_np in enumerate(masks):
#         axes[i+1].imshow(mask_np, cmap='gray')
#         axes[i+1].set_title(f"Mask {i+1}")
#         axes[i+1].axis('off')

#     plt.tight_layout()
#     plt.savefig(detect_path, bbox_inches='tight')
#     plt.close()
#     print('masked image saved')

# def save_mask(mask_np, filename):
#     mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
#     mask_image.save(filename)