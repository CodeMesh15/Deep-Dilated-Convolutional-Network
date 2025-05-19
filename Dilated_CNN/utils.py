import torchvision
import torchvision.transforms as transforms
import torch

def get_data_loaders(batch_size=32):
    transform = transforms.ToTensor()
    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True), \
           torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

def accuracy(model, loader):
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
