import torch
import torch.nn as nn

class linear_classifier(nn.Module):
    def __init__(self):
        super(linear_classifier, self).__init__()
        self.linear = nn.Linear(32*32*3,10)

    def forward(self, x):
        return self.linear(x.view(-1,3*32*32))

