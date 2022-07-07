from torch import nn


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

    def forward_once(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.sigmoid(self.bn5(self.conv5(x)))
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2