import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import utils.path_setup
from models.resnet.resnet import ResNet152


def main():
    # DataLoader 정의
    train_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean = (0.485, 0.456, 0.406), 
                            std = (0.229, 0.224, 0.2225))
    ])

    train_dataset = CIFAR10(root = './data/home/tmdals274/NNstudy/data', 
                            train = True, 
                            transform = train_transform, 
                            download = True) 

    train_dataloader = DataLoader(dataset = train_dataset, 
                                batch_size = 128, 
                                shuffle = True)

    val_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean = (0.485, 0.456, 0.406), 
                            std = (0.229, 0.224, 0.225))
    ])

    val_dataset = CIFAR10(root = './data/home/tmdals274/NNstudy/data', 
                        train = False, 
                        transform = val_transform, 
                        download = True)

    val_dataloader = DataLoader(dataset = val_dataset, 
                                batch_size = 128, 
                                shuffle = True)
    
    # 모델 정의
    model = ResNet152(10)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 로스, 옵티마이저 정의
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params = model.parameters(), 
                        lr = 0.1, 
                        momentum = 0.9, 
                        weight_decay = 1e-4)

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer = optimizer, 
                                                    gamma = 0.1)
    
    # for문 (훈련)
    epochs = 150
    best_accuracy = 0.
    total_time = 0.

    for epoch in range(epochs):
        start = time.time()
        model.train()
        epoch_loss = 0.
        
        if epoch == 99 or epoch ==124:
            lr_scheduler.step()
        
        for data in train_dataloader:
            model.zero_grad()
            image, label = data[0].to(device), data[1].to(device)
            prediction = model(image)
            loss = loss_fn(prediction, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f'\nEpoch {epoch + 1} Train Loss: {epoch_loss:.4e}')

        model.eval()
        correct = 0.
        with torch.no_grad():
            for data in val_dataloader:
                image, label = data[0].to(device), data[1].to(device)
                prediction = model(image)
                prediction = torch.argmax(prediction.data, 1)
                correct += (prediction == label).sum().item()
                accuracy = 100 * correct / len(val_dataset)

            print(f"Epoch {epoch + 1} Validation Accuracy: {accuracy:.2f}")

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            print(f"New Best Accuracy: {best_accuracy:.2f}")
            torch.save({
                'model_state_dict': model.state_dict()
            }, 
            'model.pt')
        
        end = time.time()
        total_time += end - start
        print(f"Total Time for Epoch {epoch + 1}:  {end - start}")

    # 최종 결과 show
    print(f"Total Time: {total_time:.6f}")


if __name__ == '__main__':
    main()