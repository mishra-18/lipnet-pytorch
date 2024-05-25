import torch
import torch.nn as nn

class Conv3DLSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super(Conv3DLSTMModel, self).__init__()

        self.conv1 = nn.Conv3d(1, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv3 = nn.Conv3d(256, 75, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.lstm1 = nn.LSTM(input_size=75 * 5 * 17, hidden_size=128,
                              batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)

        self.lstm2 = nn.LSTM(input_size=128 * 2, hidden_size=128,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)

        self.dense = nn.Linear(128 * 2, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Apply the sequence of conv, relu activations and max pooling
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        # Flatten the dimensions other than batch and sequence length (depth)
        batch_size, _, D, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # Swap the depth and channel dimensions
        x = x.reshape(batch_size, D, -1)  # Flatten the spatial dimensions
        # Bidirectional LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # To apply the dense layer, we need to consider only the last output of the sequence.
        x = self.dense(x)

        return x
    
class Conv3DLSTMModelMini(nn.Module):
    def __init__(self, vocab_size):
        super(Conv3DLSTMModelMini, self).__init__()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.lstm1 = nn.LSTM(input_size=128 * 11 * 35, hidden_size=64,
                              batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)

        self.dense = nn.Linear(64 * 2, vocab_size)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        batch_size, _, D, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  
        x = x.reshape(batch_size, D, -1)  # Flatten the spatial dimensions

        # Bidirectional LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x = self.dense(x)

        return x

class LipNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv3 = nn.Conv3d(64, 75, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.lstm1 = nn.LSTM(input_size=75 * 5 * 17, hidden_size=256,
                              batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)

        self.dense = nn.Linear(256 * 2, vocab_size)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))

        batch_size, _, D, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  
        x = x.reshape(batch_size, D, -1) 
        
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x = self.dense(x)

        return x

