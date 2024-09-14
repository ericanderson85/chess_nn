import torch.nn as nn


class ChessNeuralNetwork(nn.Module):
    def __init__(self):
        super(ChessNeuralNetwork, self).__init__()

        # First convolutional layer:
        self.conv1 = nn.Conv2d(
            in_channels=18, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Second convolutional layer:
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Third convolutional layer:
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Fourth convolutional layer:
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Global Average Pooling layer to reduce each feature map to a single value
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected linear layer to map the pooled features to a single output value
        self.fc = nn.Linear(512, 1)

        # Activation function
        self.relu = nn.ReLU()

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Apply first convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Apply second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Apply third convolutional layer
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # Apply fourth convolutional layer
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # Apply global average pooling to reduce spatial dimensions to 1x1
        x = self.global_avg_pool(x)

        # Flatten the tensor to shape (N, 512)
        x = x.view(x.size(0), -1)

        # Apply dropout for regularization
        x = self.dropout(x)

        # Pass through the fully connected layer to get the final evaluation
        x = self.fc(x)

        # The output is a single value per input representing the evaluation
        return x
