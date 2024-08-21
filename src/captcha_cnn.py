import torch
import torch.nn as nn

class CaptchaCNN(nn.Module):
    def __init__(self, params):
        super(CaptchaCNN, self).__init__()
        self.params = params
        self.conv1 = nn.Conv2d(1, self.params["num_filters"], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(self.params["dropout_rate"])
        # 10*10 data after pooling is 5*5, 36 output neurons for 26 letters + 10 digits
        self.fc1 = nn.Linear(self.params["num_filters"] * 5 * 5, 36) 

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, self.params["num_filters"] * 5 * 5)
        x = self.dropout(x)
        x = self.fc1(x)
        return x