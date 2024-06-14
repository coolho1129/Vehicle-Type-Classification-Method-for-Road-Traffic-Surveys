import pandas as pd
import os
from natsort import natsorted
from sklearn.metrics import accuracy_score, f1_score
from pycaret.classification import *
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

"""
용도: 차량 12종 분류 정확도 측정 및 혼동행렬 그리기

실행 방법 
vehicle_classifier 폴더로 이동 후 터미널에
trailer_classifier_predict.py입력

코드실행 결과
./ml/model 폴더에 저장된 모델을 이용하여 트레일러 분류 진행
결과 값을 csv파일로 저장 
(저장 경로: ./result/ml_based or ./result/rule_based )

각 데이터 셋에 대한 정확률 출력

혼동행렬 이미지 저장(저장 경로: ./result/ml_based)
"""


def draw_conf_matrix(df, savepath, mode):
    # 혼동 행렬 생성
    labels = sorted(df["label"].unique())
    conf_matrix = confusion_matrix(df["label"], df["pred"])
    conf_matrix_norm = (
        conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
    ) * 100

    # Heatmap으로 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_norm, annot=True, fmt=".1f", cmap="Blues")
    plt.xlabel("predict")
    plt.ylabel("ground_truth")
    plt.title("confuse_matrix")

    # x, y축 레이블을 1부터 시작하도록 조정
    plt.xticks(
        ticks=np.arange(len(labels)) + 0.5, labels=[str(label) for label in labels]
    )
    plt.yticks(
        ticks=np.arange(len(labels)) + 0.5, labels=[str(label) for label in labels]
    )

    plt.savefig(f"{savepath}/{mode}_confusion_matrix.png")  # 이미지로 저장
    print("save confusion_matrix.png")


def print_acc(df, save_folder, mode):

    labels = df.iloc[:, 1]
    preds = df.iloc[:, 2]
    accuracy = accuracy_score(labels, preds)
    accuracy *= 100
    f1 = f1_score(labels, preds, average="macro")
    f1 *= 100
    print(f"Data Set: {mode}")

    classNames = [
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "t7",
        "t8",
        "t9",
        "t10",
        "t11",
        "t12",
    ]
    classes = [0] * len(classNames)
    correct = [0] * len(classNames)
    res = []
    for label, pred in zip(labels, preds):
        idx = int(label) - 1
        classes[idx] += 1

        if label == pred:
            correct[idx] += 1

    for i in range(len(classNames)):
        class_accuracy = correct[i] / classes[i] if classes[i] != 0 else 0.0
        class_accuracy *= 100
        res.append(f"Accuracy of class {classNames[i]}: {class_accuracy:.2f}")
        print(res[i])

    trailer_acc = 0

    for i in range(7, 12):
        trailer_acc += correct[i] / classes[i] if classes[i] != 0 else 0.0

    trailer_acc = trailer_acc / 5
    trailer_acc *= 100
    print("\nTrailer Accuracy")
    print(f"{trailer_acc:.2f}\n")

    print(f"Total Accuracy:{accuracy:.2f}")
    print(f"Total f1 score:{f1:.2f}")
    print("_______________________________________\n")

    os.makedirs(save_folder, exist_ok=True)
    save_path = f"{save_folder}/{mode}_result.csv"
    df.to_csv(save_path, index=False)


def rule_base(savepath=None):
    def rule_predict(csv_file):
        csv_name = csv_file.split("/")[-1]
        types = csv_name.split("_")[1]
        df = pd.read_csv(csv_file)
        types = types.lower()
        names = df.iloc[:, 0].tolist()
        gts = []
        preds = []

        for row in df.itertuples():
            cnt = row.wheel
            if types == "semi":
                label = 0
                if cnt == 4:
                    label = 8
                elif cnt == 5:
                    label = 10
                elif cnt >= 6:
                    label = 12

            elif types == "full":
                cnt = row.wheel
                label = 0
                if cnt == 4:
                    label = 9
                elif cnt == 5:
                    label = 11

            preds.append(label)

        gts = df["label"].tolist()

        res_df = pd.DataFrame({"img": names, "label": gts, "pred": preds})

        return res_df

    if not savepath:
        savepath = "./result/rule_based/"

    for (full, semi), cls_pred in zip(wheels, cls_preds):
        mode = full.split("_")[0]
        full_csv = f"{wheels_path}/{full}"
        semi_csv = f"{wheels_path}/{semi}"
        cls_pred_csv = f"{cls_pred_path}/{cls_pred}"
        pred_df = pd.read_csv(cls_pred_csv)
        one2Sev_pred_df = pred_df[pred_df.iloc[:, 1] < 8]
        one2Sev_pred_df = one2Sev_pred_df.drop(columns=["class12"])

        full_rough_rule_df = rule_predict(full_csv)
        semi_rough_rule_df = rule_predict(semi_csv)
        result_df = pd.concat([one2Sev_pred_df, full_rough_rule_df, semi_rough_rule_df])
        print_acc(result_df, savepath, mode)


def ml_base(data, path, fullmodel, semimodel, savepath=None):
    def ml_predict(df, model_path):
        result_df = df.drop(columns=["wheel"])
        pred_df = df.drop(columns=["label"])

        final_model = load_model(model_path)

        pred_label = predict_model(final_model, pred_df)["prediction_label"].to_list()
        result_df["pred"] = pred_label

        return result_df

    if not savepath:
        savepath = "./result/ml_based/"

    for (full, semi), cls_pred in zip(data, cls_preds):
        mode = full.split("_")[0]
        full_csv = f"{path}/{full}"
        semi_csv = f"{path}/{semi}"
        cls_pred_csv = f"{cls_pred_path}/{cls_pred}"

        full_df = pd.read_csv(full_csv)
        semi_df = pd.read_csv(semi_csv)
        other_pred_df = pd.read_csv(cls_pred_csv)
        other_pred_df = other_pred_df[other_pred_df.iloc[:, 1] < 8]

        print(mode)
        print(f"full-trailer-classification")
        full_pred_df = ml_predict(full_df, fullmodel)
        print("full-trailer-classification done")
        print("semi-trailer-classification")
        semi_pred_df = ml_predict(semi_df, semimodel)
        print("semi-trailer-classificationdone\n")

        pred_df = pd.concat([other_pred_df, full_pred_df, semi_pred_df])
        pred_df = pred_df[["img", "label", "pred"]]
        sorted_img = natsorted(pred_df["img"].tolist())
        result_df = pred_df.set_index("img").loc[sorted_img].reset_index()

        draw_conf_matrix(result_df, savepath, mode)
        print_acc(result_df, savepath, mode)


wheels_path = "./detect/csv/"
cls_pred_path = "./init_classification/csv"

wheels = [
    ("train_Full_wheels.csv", "train_Semi_wheels.csv"),
    ("valid_Full_wheels.csv", "valid_Semi_wheels.csv"),
    ("test_Full_wheels.csv", "test_Semi_wheels.csv"),
]
cls_preds = ["train_pred.csv", "valid_pred.csv", "test_pred.csv"]


"""
용도: 차량 12종 분류 정확도 측정 및 혼동행렬 그리기

실행 방법 
vehicle_classifier 폴더로 이동 후 터미널에
trailer_classifier_predict.py입력

코드실행 결과
./ml/model 폴더에 저장된 모델을 이용하여 트레일러 분류 진행
결과 값을 csv파일로 저장 
(저장 경로: ./result/ml_based or ./result/rule_based )

각 데이터 셋에 대한 정확률 출력

혼동행렬 이미지 저장(저장 경로: ./result/ml_based)

"""


def main():
    print("rule-based")
    print("_______________________________________")
    rule_base()

    print("ml-based")
    print("_______________________________________")
    full_model = "./ml/model/full_model"
    semi_model = "./ml/model/semi_model"
    ml_base(wheels, wheels_path, full_model, semi_model)


if __name__ == "__main__":
    main()
