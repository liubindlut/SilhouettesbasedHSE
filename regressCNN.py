import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class conv1_1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(conv1_1, self).__init__()

        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class MLP_liu(nn.Module):
    def __init__(self, inplanes, outplanes, size_kernel=3, size_padding=1, size_pooling=3):
        super(MLP_liu, self).__init__()

        self.conv = nn.Conv1d(inplanes, outplanes, kernel_size=size_kernel, padding=size_padding, bias=False)
        self.bn = nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU(inplace=True)

        self.pooling = nn.MaxPool1d(size_pooling, size_pooling)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pooling(out)

        return out


class RegressionPCA(nn.Module):
    def __init__(self, len_out):
        super(RegressionPCA, self).__init__()

        hidden1 = 2*512 
        hidden2 = 512
        num_keypoints = 8 # (num of points: num_keypoints) (648, 8) Namely, (divide 81)
        c1, c2, c3, c4, c5 = (64, 128, 192, 192, 144)
        #c1, c2, c3, c4, c5 = (64, 96, 128, 160, 128)

        # front
        self.layer11 = conv1_1(2, c1)
        self.layer12 = MLP_liu(c1, c2)
        self.layer13 = MLP_liu(c2, c3)
        self.layer14 = MLP_liu(c3, c4)
        # side
        self.layer21 = conv1_1(2, c1)
        self.layer22 = MLP_liu(c1, c2)
        self.layer23 = MLP_liu(c2, c3)
        self.layer24 = MLP_liu(c3, c4)
        # merge
        self.layer31 = MLP_liu(c1, c2)  
        self.layer32 = MLP_liu(c2, c3)
        self.layer33 = MLP_liu(c3, c4)
        self.layer34 = MLP_liu(c4, c5)

        self.fc = nn.Sequential(
            nn.Linear(num_keypoints*c5, hidden1), 
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, len_out)                             
        )

    def forward(self, input1, input2): # feed data

        f1 = self.layer11(input1)
        s1 = self.layer21(input2)
        fs1 = self.layer31(f1+s1)

        f2 = self.layer12(f1)
        s2 = self.layer22(s1)
        fs2 = self.layer32(f2+s2+fs1)

        f3 = self.layer13(f2)
        s3 = self.layer23(s2)
        fs3 = self.layer33(f3+s3+fs2)

        f4 = self.layer14(f3)
        s4 = self.layer24(s3)
        fs4 = self.layer34(f4+s4+fs3)

        fs5 = fs4.view(fs4.shape[0], -1)
        out = self.fc(fs5)

        return out   

if __name__ == '__main__':
    len_out = 9
    model = RegressionPCA(len_out)
    print(model)
