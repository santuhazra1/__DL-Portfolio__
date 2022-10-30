import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, get_norm, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = get_norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = get_norm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                get_norm(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_batchnorm=False, use_groupnorm=False, use_layernorm=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_batchnorm = use_batchnorm
        self.use_groupnorm = use_groupnorm
        self.use_layernorm = use_layernorm

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = self._get_norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self._get_norm, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _get_norm(self, inp_feature):
        if self.use_batchnorm:
            return nn.BatchNorm2d(inp_feature)
        elif self.use_groupnorm:
            return nn.GroupNorm(num_groups=2, num_channels=inp_feature)
        elif self.use_layernorm:
            return nn.GroupNorm(num_groups=1, num_channels= inp_feature)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(use_batchnorm=False, use_groupnorm=False, use_layernorm=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], use_batchnorm=use_batchnorm, use_groupnorm=use_groupnorm, use_layernorm=use_layernorm)


def ResNet34(use_batchnorm=False, use_groupnorm=False, use_layernorm=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], use_batchnorm=use_batchnorm, use_groupnorm=use_groupnorm, use_layernorm=use_layernorm)

# class LayerNorm(nn.Module):
#     def __init__(self, num_features, eps=1e-5, affine=True):
#         super(LayerNorm, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps

#         if self.affine:
#             self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x):
#         shape = [-1] + [1] * (x.dim() - 1)
#         mean = x.view(x.size(0), -1).mean(1).view(*shape)
#         std = x.view(x.size(0), -1).std(1).view(*shape)
#         x = (x - mean) / (std + self.eps)

#         if self.affine:
#             shape = [1, -1] + [1] * (x.dim() - 2)
#             x = x * self.gamma.view(*shape) + self.beta.view(*shape)
#         return x

# SEED = 69
# use_cuda = torch.cuda.is_available()


# def test():
#     use_cuda = torch.cuda.is_available()
#     SEED = 42
#     torch.manual_seed(SEED)
#     if use_cuda:
#         torch.cuda.manual_seed(SEED)
#     device = torch.device("cuda" if use_cuda else "cpu")

#     model = ResNet18(use_layernorm=True).to(device)
#     y = model(torch.randn(1, 3, 32, 32).to(device))
#     print(y.size())
#     summary(model, input_size=(3, 32, 32))

# test()