# This is a sample Python script.

import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from bagnetmaster.bagnets.pytorchnet import bagnet33


def custom_collate_fn(batch):
    selected_classes = [444, 22, 817, 836, 339, 404, 403, 834]
    class_map = {original: new for new, original in enumerate(selected_classes)}
    filtered_batch = [(img, class_map[label]) for img, label in batch if label in class_map]
    if not filtered_batch:
        return None, None
    images, labels = zip(*filtered_batch)
    return torch.stack(images), torch.tensor(labels)


def retrain_resnet():
    batch_size = 64
    num_epochs = 15

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageNet(root='C:\\Users\\Lumpe\\Pictures\\Imagenet', split='train', transform=transform)
    val_dataset = datasets.ImageNet(root='C:\\Users\\Lumpe\\Pictures\\Imagenet', split='val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                            collate_fn=custom_collate_fn)

    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)

    for param in model.fc.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        print("CUDA is available. Number of CUDA devices:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    print("Model " + str(device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.008)

    best_accuracy = 0

    for epoch in range(num_epochs):
        if epoch > 5:
            for param in model.layer4.parameters():
                param.requires_grad = False

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            if inputs is None:
                continue

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if inputs is None:
                    continue

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'resnet50_acc_' + str(accuracy) + '.pth')


def run_comparison_resnet():
    bagnet = models.resnet50(pretrained=True)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    bagnet.eval()

    with open('mirc_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    overview = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    picture_list = (overview['class'] + '\\patch_id' + overview['id'].astype(str)).tolist()
    #folder_list = ["01_flipvertical_mircs", "04_inverse_mircs", "06_texture_mircs"]
    folder_list = ["color_mircs"]
    result_list = []
    pre = "C:\\Users\\Lumpe\\Synced\\_CogCoVI\\coloring\\pythonProject3\\"

    for f in folder_list:
        for p in picture_list:
            img = Image.open(pre + f + "\\" + p + ".png")
            img_t = transform(img)
            batch_t = img_t.unsqueeze(0)
            out = bagnet(batch_t)
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
    #df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_bagnet_results.csv", sep=",", index=False)
    df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_bagnet_results_originals.csv", sep=",", index=False)


def run_comparison_retrained2():
    retrained = torchvision.models.resnet50(pretrained=False)
    num_ftrs = retrained.fc.in_features
    retrained.fc = nn.Linear(num_ftrs, 8)
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
    #folder_list = ["01_flipvertical_mircs", "04_inverse_mircs", "06_texture_mircs"]
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
    #df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_results.csv", sep=",", index=False)
    df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_results_originals.csv", sep=",", index=False)


def run_comparison_model_human():
    retrained = torchvision.models.resnet50(pretrained=False)
    num_ftrs = retrained.fc.in_features
    retrained.fc = nn.Linear(num_ftrs, 8)
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
    human_average = overview['Average'].tolist()
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
    x = np.arange(24)  # the label locations
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
    ax.set_title('Picture-wise Comparison Human vs Model')
    ax.set_xticks(x + width, (overview["Class"] + "_" + overview["degradation"].astype(str)).tolist(), rotation=90)
    ax.legend(loc='upper right', ncols=3)
    ax.set_ylim(0, 110)

    plt.show()


def run_comparison_retrained():
    retrained = torchvision.models.resnet50(pretrained=False)
    num_ftrs = retrained.fc.in_features
    retrained.fc = nn.Linear(num_ftrs, 8)
    retrained.load_state_dict(torch.load("resnet50_acc_94.5.pth"))
    retrained.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    with open('mirc_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    overview = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
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

    column_names = ["effect", "real_class",]
    for c in classes:
        column_names.extend([c])
        column_names.extend([c + "_confidence"])

    df = pd.DataFrame(result_list, columns=column_names)
    df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_results_color_full.csv", sep=",", index=False)


def combine_data():
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
    #degraded = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_results.csv", sep=",")
    degraded = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_bagnet_results.csv", sep=",")
    #original = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_results_originals.csv", sep=",")
    original = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\refitted_bagnet_results_originals.csv", sep=",")
    degraded = degraded[["effect", "real_class", "id", "guessed_class", "max_confidence", "Correct"]]
    original = original[["real_class", "id", "guessed_class", "max_confidence", "Correct"]]
    original.rename(columns={"guessed_class": "original_guess", "max_confidence": "original_confidence", "Correct": "original_correct"}, inplace=True)
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

    for c in ["bike", "car", "eagle", "glasses", "horse", "plane", "ship", "suit" ]:
        current = degraded[degraded["real_class"] == c]
        aggregated = current.groupby('id').sum().reset_index()
        aggregated["Correct"] = aggregated["Correct"]
        aggregated["original_correct"] = aggregated["original_correct"]
        #aggregated["Correct"] = aggregated["Correct"].round().astype(int)
        #aggregated["original_correct"] = aggregated["original_correct"].round().astype(int)

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

if __name__ == '__main__':
    run_comparison_model_human()
    #run_comparison_resnet()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


'''
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dataset = CustomImageDataset(root_dir='C:\\Users\\Lumpe\\Pictures\\Imagenet', transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    '''
