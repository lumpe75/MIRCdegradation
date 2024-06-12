import torch
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


def custom_collate_fn(batch):
    """
    Applies a class filter to a batch to retrain a network
    """
    selected_classes = [444, 22, 817, 836, 339, 404, 403, 834]  # the classes to be trained on
    class_map = {original: new for new, original in enumerate(selected_classes)}
    filtered_batch = [(img, class_map[label]) for img, label in batch if label in class_map]
    if not filtered_batch:
        return None, None
    images, labels = zip(*filtered_batch)
    return torch.stack(images), torch.tensor(labels)


def retrain_resnet():
    """
    Retrain a Resnet 50 to the 8 classes specified in 'custom_collate_fn'
    """
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