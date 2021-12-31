from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcMarginProduct(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.s            = s
        self.m            = m
        self.easy_margin  = easy_margin

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m


    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        #cos(a + self.m) = cos(a)*cos(self.m) - sin(a)*sin(self.m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # m 만큼 더해주는 이유는 더 패널티를 주게하기 위함인데
            # m 을 더해서 180도가 될 때까지만 그렇게 하고
            # 
            # 넘어섰을 때는 
            # ???
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)


        # torch.where(out_i = {x_i if condition_i else y_i)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output



def main():
    import numpy as np
    metric_fc = ArcMarginProduct(512, 10000, s=30, m=0.5,
                                 easy_margin=False)
    print(metric_fc.weight)
    input = np.random.rand(1, 512)
    label = np.array([[3]])
    input = torch.from_numpy(input).type(torch.FloatTensor)
    label = torch.from_numpy(label).type(torch.FloatTensor)
    metric_fc.forward(input, label)


if __name__ == '__main__':
    main()
