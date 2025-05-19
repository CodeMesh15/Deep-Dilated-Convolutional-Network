import torch.nn as nn

class DilatedCNN(nn.Module):
    def __init__(self):
        super(DilatedCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 CIFAR-10 classes
        )

    def forward(self, x):
        return self.net(x)
