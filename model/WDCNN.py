import torch
from torch import nn



class WDCNN(nn.Module):
    
    def __init__(self, input_channel, output_channel):
        super(WDCNN,self).__init__()
        # padding = (kernel_size - stride) // 2
        # output_length = floor((input_length + 2×padding - kernel_size) / stride) + 1

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channel,out_channels=16,kernel_size=64,stride=16,padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.Dropout(0.05),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )    
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )   
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.05),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )   
        self.layer4 = nn.Sequential(
            nn.Conv1d(64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.05),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )   
        self.layer5 = nn.Sequential(
            nn.Conv1d(64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256,100),
            nn.ReLU(),
            # nn.Dropout(0.05),
            nn.Linear(100,output_channel)
        )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
