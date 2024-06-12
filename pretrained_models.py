import pandas as pd
from torchvision import models
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import torch


def run_comparison_alex_net():
    """
    Scores the padded degraded MIRCs with a pretrained Alexnet
    """
    alex_net = models.alexnet(pretrained=True)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    alex_net.eval()
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    overview = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    picture_list = (overview['class'] + '_' + overview['id'].astype(str)).tolist()
    folder_list = ["07_inverse_padded", "08_flip_padded", "09_texture_padded"]
    result_list = []

    for f in folder_list:
        for p in picture_list:
            img = Image.open(f + "\\" + p + ".png")
            img_t = transform(img)
            batch_t = torch.unsqueeze(img_t, 0)
            out = alex_net(batch_t)
            percentage = out[0]
            current = [f, p.split('_')[0], p.split('_')[1]]
            for idx, _ in enumerate(classes):
                class_name = classes[idx]
                confidence = percentage[idx].item()
                current.extend([class_name, confidence])
            result_list.append(current)

    column_names = ["effect", "real_class", "id"]
    for c in classes:
        column_names.extend([c])
        column_names.extend([c + "_confidence"])

    df = pd.DataFrame(result_list, columns=column_names)
    df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\alexnet_results.csv", sep=",", index=False)


def run_comparison_resnet():
    """
    Scores the padded degraded MIRCs with a pretrained Resnet 50
    """
    resnet = models.resnet50(pretrained=True)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    resnet.eval()
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    overview = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    picture_list = (overview['class'] + '_' + overview['id'].astype(str)).tolist()
    folder_list = ["07_inverse_padded", "08_flip_padded", "09_texture_padded"]
    result_list = []

    for f in folder_list:
        for p in picture_list:
            img = Image.open(f + "\\" + p + ".png")
            img_t = transform(img)
            batch_t = torch.unsqueeze(img_t, 0)
            out = resnet(batch_t)

            percentage = out[0]

            current = [f, p.split('_')[0], p.split('_')[1]]
            for idx, _ in enumerate(classes):
                class_name = classes[idx]
                confidence = percentage[idx].item()
                current.extend([class_name, confidence])
            result_list.append(current)

    column_names = ["effect", "real_class", "id"]
    for c in classes:
        column_names.extend([c])
        column_names.extend([c + "_confidence"])

    df = pd.DataFrame(result_list, columns=column_names)
    df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\resnet_results.csv", sep=",", index=False)


def run_comparison_rcnn():
    """
    Scores the padded degraded MIRCs with a pretrained Faster R-CNN model
    """
    rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
    rcnn.eval()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    overview = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    picture_list = (overview['class'] + '_' + overview['id'].astype(str)).tolist()
    folder_list = ["07_inverse_padded", "08_flip_padded", "09_texture_padded"]
    result_list = []

    for f in folder_list:
        for p in picture_list:
            img = Image.open(f + "\\" + p + ".png")
            img_t = transform(img)
            batch_t = torch.unsqueeze(img_t, 0)
            with torch.no_grad():
                predictions = rcnn(batch_t)

            prediction = predictions[0]

            # labels = prediction['labels']
            scores = prediction['scores']

            percentage = torch.nn.functional.softmax(scores, dim=1)[0] * 100

            current = [f, p.split('_')[0], p.split('_')[1]]
            for idx, _ in enumerate(classes):
                class_name = classes[idx]
                confidence = percentage[idx].item()
                current.extend([class_name, confidence])
            result_list.append(current)

    column_names = ["effect", "real_class", "id"]
    for c in classes:
        column_names.extend([c])
        column_names.extend([c + "_confidence"])

    df = pd.DataFrame(result_list, columns=column_names)
    df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\rcnn_results.csv", sep=",", index=False)
