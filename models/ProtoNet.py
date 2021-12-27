import torch
import torch.nn as nn


class ProtoNet(nn.Module):

    def __init__(self):
        super(ProtoNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.conv_1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.conv_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.conv_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

    def forward(self, inputs):
        conv_1 = self.conv_1(inputs)
        conv_1 = self.maxpool(conv_1)

        conv_2 = self.conv_2(conv_1)
        conv_2 = self.maxpool(conv_2)

        conv_3 = self.conv_3(conv_2)
        conv_3 = self.maxpool(conv_3)

        conv_4 = self.conv_4(conv_3)
        outputs = self.maxpool(conv_4)

        outputs = outputs.view(outputs.size(0), -1) # Flatten

        return outputs


if __name__ == "__main__":
    t = torch.randn(2, 1, 256, 256).cuda()
    n = ProtoNet().cuda()
    print(n(t).size())
