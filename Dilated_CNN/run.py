from model import DilatedCNN
from train import train
from test import evaluate

model = DilatedCNN()
train(model, epochs=5)
evaluate(model)
