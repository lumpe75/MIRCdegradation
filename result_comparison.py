import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
import torch
from torchvision.transforms import transforms
import torch.nn as nn
from torchvision import models, transforms


def run_comparison_resnet():
    """
    Scores the MIRCs with a Resnet 50, applying a softmax and choosing the most likely class for each image
    """
    resnet = models.resnet50(pretrained=True)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    resnet.eval()

    with open('mirc_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    overview = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    picture_list = (overview['class'] + '\\patch_id' + overview['id'].astype(str)).tolist()
    # folder_list = ["01_flipvertical_mircs", "04_inverse_mircs", "06_texture_mircs"]
    folder_list = ["color_mircs"]
    result_list = []
    pre = "C:\\Users\\Lumpe\\Synced\\_CogCoVI\\coloring\\pythonProject3\\"

    for f in folder_list:
        for p in picture_list:
            img = Image.open(pre + f + "\\" + p + ".png")
            img_t = transform(img)
            batch_t = img_t.unsqueeze(0)
            out = resnet(batch_t)
            out = F.softmax(out, dim=1)

            percentage = out[0]

            current = [f, p.split('_')[0].split('\\')[0], p.split('_')[1]]
            max_class = ""
            max_percentage = 0
            for idx, _ in enumerate(classes):
                class_name = classes[idx]
                confidence = percentage[idx].item()
                if confidence > max_percentage:
                    max_percentage = confidence
                    max_class = class_name
                current.extend([class_name, confidence])
            current.extend([max_class, max_percentage])
            result_list.append(current)

    column_names = ["effect", "real_class", "id"]
    for c in classes:
        column_names.extend([c])
        column_names.extend([c + "_confidence"])
    column_names.extend(["guessed_class", "max_confidence"])

    df = pd.DataFrame(result_list, columns=column_names)
    df["Correct"] = df["real_class"] == df["guessed_class"]
    # df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_resnet_results.csv",
    #          sep=",", index=False)
    df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_resnet_results_originals.csv",
              sep=",", index=False)


def run_comparison_retrained():
    """
    Scores the MIRCs with a retrained Resnet 50 to fit 8 classes, applying a softmax and choosing the most likely class
    for each image
    """
    retrained = torchvision.models.resnet50(pretrained=False)
    num_features = retrained.fc.in_features
    retrained.fc = nn.Linear(num_features, 8)
    retrained.load_state_dict(torch.load("resnet50_acc_94.5.pth"))
    retrained.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    with open('mirc_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    overview = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    picture_list = (overview['class'] + '\\patch_id' + overview['id'].astype(str)).tolist()
    # folder_list = ["01_flipvertical_mircs", "04_inverse_mircs", "06_texture_mircs"]
    folder_list = ["color_mircs"]
    result_list = []
    pre = "C:\\Users\\Lumpe\\Synced\\_CogCoVI\\coloring\\pythonProject3\\"

    for f in folder_list:
        for p in picture_list:
            img = Image.open(pre + f + "\\" + p + ".png")
            img_t = transform(img)
            batch_t = img_t.unsqueeze(0)
            out = retrained(batch_t)
            out = F.softmax(out, dim=1)

            percentage = out[0]

            current = [f, p.split('_')[0].split('\\')[0], p.split('_')[1]]
            max_class = ""
            max_percentage = 0
            for idx, _ in enumerate(classes):
                class_name = classes[idx]
                confidence = percentage[idx].item()
                if confidence > max_percentage:
                    max_percentage = confidence
                    max_class = class_name
                current.extend([class_name, confidence])
            current.extend([max_class, max_percentage])
            result_list.append(current)

    column_names = ["effect", "real_class", "id"]
    for c in classes:
        column_names.extend([c])
        column_names.extend([c + "_confidence"])
    column_names.extend(["guessed_class", "max_confidence"])

    df = pd.DataFrame(result_list, columns=column_names)
    df["Correct"] = df["real_class"] == df["guessed_class"]
    # df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_results.csv", sep=",", index=False)
    df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_results_originals.csv", sep=",", index=False)


def run_comparison_model_human():
    """
    Visualizes the recognition difference between a refitted Resnet 50 and the human participants over each individual
    image
    """
    retrained = torchvision.models.resnet50(pretrained=False)
    num_features = retrained.fc.in_features
    retrained.fc = nn.Linear(num_features, 8)
    retrained.load_state_dict(torch.load("resnet50_acc_94.5.pth"))
    retrained.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    with open('mirc_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    overview = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\results\\participant_results.csv", sep=",")
    overview['path'] = overview['degradation'].replace({'flip': '01_flipvertical_mircs',
                                                        'inverse': '04_inverse_mircs',
                                                        "texture": "06_texture_mircs"}, regex=True)
    picture_list = (overview['path'] + "\\" + overview["Class"] + '\\' + overview['OldID'].astype(str)).tolist()
    class_list = overview['Class'].tolist()
    pre = "C:\\Users\\Lumpe\\Synced\\_CogCoVI\\coloring\\pythonProject3\\"
    result_list = []
    for i, p in enumerate(picture_list):
        img = Image.open(pre + p + ".png")
        img_t = transform(img)
        batch_t = img_t.unsqueeze(0)
        out = retrained(batch_t)
        out = F.softmax(out, dim=1)
        percentage = out[0]

        real_percentage = 0
        for idx, _ in enumerate(classes):
            if classes[idx] == class_list[i]:
                real_percentage = percentage[idx].item()
        result_list.append(real_percentage)
    overview["model_guess"] = result_list

    graph_values = {
        "Human": list((overview["Average"] * 100).round().astype(int)),
        "Model": list((overview["model_guess"] * 100).round().astype(int))
    }
    x = np.arange(24)
    width = 0.25
    multiplier = 1
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in graph_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Recognition in percent')
    ax.set_title('Picture-wise Comparison Human vs Model')
    ax.set_xticks(x + width, (overview["Class"] + "_" + overview["degradation"].astype(str)).tolist(), rotation=90)
    ax.legend(loc='upper right', ncols=3)
    ax.set_ylim(0, 110)

    plt.show()


def run_comparison_retrained_old():
    """
    Scores the original images with a retrained Resnet 50 to fit 8 classes, applying a softmax and choosing the most
    likely class for each image
    """
    retrained = torchvision.models.resnet50(pretrained=False)
    num_features = retrained.fc.in_features
    retrained.fc = nn.Linear(num_features, 8)
    retrained.load_state_dict(torch.load("resnet50_acc_94.5.pth"))
    retrained.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    with open('mirc_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    picture_list = [
        "bike_edited400px.png",
        "car_edited400px.png",
        "eagle_edited400px.png",
        "glasses_edited400px.png",
        "horse_edited400px.png",
        "plane_edited400px.png",
        "ship_edited400px.png",
        "suit_edited400px.png"
    ]
    result_list = []
    pre = "C:\\Users\\Lumpe\\Synced\\_CogCoVI\\coloring\\pythonProject3\\_coloredFull400px\\"

    for p in picture_list:
        img = Image.open(pre + p)
        img_t = transform(img)
        batch_t = img_t.unsqueeze(0)
        out = retrained(batch_t)
        out = F.softmax(out, dim=1)

        percentage = out[0]

        current = [f, p.split('_')[0]]
        for idx, _ in enumerate(classes):
            class_name = classes[idx]
            confidence = percentage[idx].item()
            current.extend([class_name, confidence])
            result_list.append(current)

    column_names = ["effect", "real_class", ]
    for c in classes:
        column_names.extend([c])
        column_names.extend([c + "_confidence"])

    df = pd.DataFrame(result_list, columns=column_names)
    df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_results_color_full.csv", sep=",", index=False)


def draw_comparison_class_and_effect_wise():
    """
    Draws a graph for each class and each effect, comparing the recognition of the original MIRC with the degraded MIRC
    """
    class_dict = {
        "bike": 21,
        "car": 3,
        "eagle": 12,
        "glasses": 10,
        "horse": 16,
        "plane": 19,
        "ship": 15,
        "suit": 9
    }
    # degraded = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_results.csv", sep=",")
    degraded = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\results\\refitted_resnet_results.csv", sep=",")
    # original = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_results_originals.csv", sep=",")
    original = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\results\\refitted_resnet_results_originals.csv",
                           sep=",")
    degraded = degraded[["effect", "real_class", "id", "guessed_class", "max_confidence", "Correct"]]
    original = original[["real_class", "id", "guessed_class", "max_confidence", "Correct"]]
    original.rename(columns={
        "guessed_class": "original_guess",
        "max_confidence": "original_confidence",
        "Correct": "original_correct"
    }, inplace=True)
    degraded = degraded.merge(original, on=["real_class", "id"], how="outer")
    for e in ["01_flipvertical_mircs", "04_inverse_mircs", "06_texture_mircs"]:
        current = degraded[degraded["effect"] == e]
        aggregated = current.groupby('real_class').sum().reset_index()
        aggregated["divisor"] = aggregated["real_class"].map(class_dict)
        aggregated["Correct"] = aggregated["Correct"] * 100 / aggregated["divisor"]
        aggregated["original_correct"] = aggregated["original_correct"] * 100 / aggregated["divisor"]
        aggregated["Correct"] = aggregated["Correct"].round().astype(int)
        aggregated["original_correct"] = aggregated["original_correct"].round().astype(int)

        graph_values = {
            "Original Mircs": list(aggregated["original_correct"]),
            "Degraded Mircs": list(aggregated["Correct"])
        }

        x = np.arange(8)  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 1
        fig, ax = plt.subplots(layout='constrained')
        for attribute, measurement in graph_values.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Recognition in percent')
        ax.set_title('Recognition for Effect ' + e.split("_")[1])
        ax.set_xticks(x + width, aggregated["real_class"])
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylim(0, 100)

        plt.show()

    for c in ["bike", "car", "eagle", "glasses", "horse", "plane", "ship", "suit"]:
        current = degraded[degraded["real_class"] == c]
        aggregated = current.groupby('id').sum().reset_index()
        aggregated["Correct"] = aggregated["Correct"]
        aggregated["original_correct"] = aggregated["original_correct"]
        # aggregated["Correct"] = aggregated["Correct"].round().astype(int)
        # aggregated["original_correct"] = aggregated["original_correct"].round().astype(int)

        graph_values = {
            "Original Mircs": list(aggregated["original_correct"]),
            "Degraded Mircs": list(aggregated["Correct"])
        }

        x = np.arange(len(aggregated))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 1
        fig, ax = plt.subplots(layout='constrained')
        for attribute, measurement in graph_values.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Recognition in total numbers')
        ax.set_title('Recognition for picture ' + c)
        ax.set_xticks(x + width, aggregated["id"])
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylim(0, 3.1)

        plt.show()


if __name__ == "__main__":
    draw_comparison_class_and_effect_wise()
