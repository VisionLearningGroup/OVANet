from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable
from .resnet import resnet50_moco
import pretrainedmodels
from .efficientnet import *
#from .san import san

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x,lambd):
        ctx.save_for_backward(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd=ctx.saved_tensors[0]
        return grad_output.neg()*lambd, None
def grad_reverse(x,lambd=1.0):
    return GradReverse.apply(x, Variable(torch.ones(1)*lambd).cuda())

def grad_reverse_amp(x, lambd=1.0):
    return GradReverse.apply(x, Variable(torch.ones(1).cuda() * lambd).half())


class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, top=False):
        super(ResBase, self).__init__()
        self.dim = 2048
        self.top = top
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet18_swsl':
            model_ft = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')
            self.dim = 512
        if option == 'resnet18_ssl':
            model_ft = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_ssl')
            self.dim = 512
        if option == 'resnet34':
            model_ft = models.resnet34(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet50_augmix':
            model_ft = models.resnet50(pretrained=pret)
            checkpoint = torch.load("/research/masaito/imagenet_models/checkpoint.pth.tar")
            dicts = checkpoint['state_dict']
            new_dict = {}
            for key in dicts.keys():
                new_dict[key.replace("module.", "")] = dicts[key]
            model_ft.load_state_dict(new_dict)
        if option == 'resnet50_ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        if option == 'resnet50_swsl':
            model_ft = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
        if option == 'resnet50_ssl':
            model_ft = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_ssl')

        if option == 'resnet50_moco':
            model_ft = resnet50_moco(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)

        if top:
            self.features = model_ft
        else:
            mod = list(model_ft.children())
            mod.pop()
            self.features = nn.Sequential(*mod)


    def forward(self, x):
        x = self.features(x)
        if self.top:
            return x
        else:
            x = x.view(x.size(0), self.dim)
            return x

class ShuffleBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, top=False):
        super(ShuffleBase, self).__init__()
        self.dim = 1024
        self.top = top
        #if option == 'shufflenet':
        model_ft = models.shufflenet_v2_x0_5(pretrained=pret)
        if top:
            self.features = model_ft
        else:
            mod = list(model_ft.children())
            mod.pop()
            self.features = nn.Sequential(*mod)
    def forward(self, x):
        x = self.features(x)
        if self.top:
            return x
        else:
            x = x.mean([2, 3])
            x = x.view(x.size(0), self.dim)
            return x


class WideRes(nn.Module):
    def __init__(self, option='resnet50', pret=True, top=False):
        super(WideRes, self).__init__()
        self.dim = 2048
        self.top = top
        if option == 'wide_resnet50':
            model_ft = models.wide_resnet50_2(pretrained=pret)
        if option == 'wide_resnet101':
            model_ft = models.wide_resnet101_2(pretrained=pret)

        if top:
            self.features = model_ft
        else:
            mod = list(model_ft.children())
            mod.pop()
            self.features = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        if self.top:
            return x
        else:
            x = x.view(x.size(0), self.dim)
            return x


class IncepionBase(nn.Module):
    def __init__(self, option='inception_v3', pret=True, top=False):
        super(IncepionBase, self).__init__()
        self.dim = 1024
        self.top = top
        print(option)
        if option == 'inception_v3':
            model_ft = models.inception_v3(pretrained=True)#torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
        if option == 'inception_google':
            model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)

        if top:
            self.features = model_ft
            #import pdb
            #pdb.set_trace()
        else:
            mod = list(model_ft.children())
            mod.pop()
            #import pdb
            #pdb.set_trace()
            self.features = nn.Sequential(*mod)

    def forward(self, x):
        #import pdb
        #pdb.set_trace
        if self.top:
            x = self.features(x)
            #import pdb
            #pdb.set_trace()
            return x
        else:
            x = self.features(x)
            x = x.view(x.size(0), self.dim)
            return x


class EfficientBase(nn.Module):
    def __init__(self, option='inception_v', pret=True, top=False):
        super(EfficientBase, self).__init__()
        self.dim = 2048
        self.top = top
        if option == 'efficient-b0':
            model_ft = efficientnet_b0(pretrained=True)
        if option == 'efficient-b4':
            model_ft = efficientnet_b4(pretrained=True)
        if top:
            self.features = model_ft
        else:
            mod = torch.nn.ModuleList(model_ft.children())[:-1]
            #import pdb
            #pdb.set_trace()
            #mod.pop()
            self.features = nn.Sequential(*mod)

    def forward(self, x):

        if self.top:
            x = self.features(x)
            return x
        else:

            #import pdb
            #pdb.set_trace()
            x = self.features(x)
            x = F.relu(x, inplace=True)
            #import pdb
            #pdb.set_trace()
            x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
            return x

class SEBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, top=False):
        super(SEBase, self).__init__()
        self.dim = 2048
        self.top = top
        if option =='se_resnet50':
            model_ft = pretrainedmodels.se_resnet50(num_classes=1000, pretrained='imagenet')
        if option == 'se_resnet101':
            model_ft = pretrainedmodels.se_resnet101(num_classes=1000, pretrained='imagenet')
        if option == 'se_resnet152':
            model_ft = pretrainedmodels.se_resnet152(num_classes=1000, pretrained='imagenet')
        #mod = list(model_ft.children())
        #mod.pop()
        if top:
            self.features = model_ft
        else:
            mod = torch.nn.ModuleList(model_ft.children())[:-1]
            self.features = nn.Sequential(*mod)


    def forward(self, x):
        x = self.features(x)
        if self.top:
            return x
        else:
            x = x.mean([2, 3])
            return x


class ResBase_v2(nn.Module):
    def __init__(self, option='resnet50', pret=True):
        super(ResBase_v2, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)
        self.fc0 = nn.Linear(self.dim, 512)
        self.bn0 = nn.BatchNorm1d(512)

    def forward(self, x):
        #x = self.in0(x)
        x = self.features(x)
        x = x.view(x.size(0), self.dim)
        x = self.bn0(self.fc0(x))
        return x


class VGGBase(nn.Module):
    def __init__(self, option='vgg', pret=True, no_pool=False, top=False):
        super(VGGBase, self).__init__()
        self.dim = 2048
        self.no_pool = no_pool
        self.top = top

        if option =='vgg11_bn':
            vgg16=models.vgg11_bn(pretrained=pret)
        elif option == 'vgg11':
            vgg16 = models.vgg11(pretrained=pret)
        elif option == 'vgg13':
            vgg16 = models.vgg13(pretrained=pret)
        elif option == 'vgg13_bn':
            vgg16 = models.vgg13_bn(pretrained=pret)
        elif option == "vgg16":
            vgg16 = models.vgg16(pretrained=pret)
        elif option == "vgg16_bn":
            vgg16 = models.vgg16_bn(pretrained=pret)
        elif option == "vgg19":
            vgg16 = models.vgg19(pretrained=pret)
        elif option == "vgg19_bn":
            vgg16 = models.vgg19_bn(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier._modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features._modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))
        if self.top:
            self.vgg = vgg16

    def forward(self, x, source=True,target=False):
        if self.top:
            x = self.vgg(x)
            return x
        else:
            x = self.features(x)
            x = x.view(x.size(0), 7 * 7 * 512)
            x = self.classifier(x)
            return x

class DenseBase(nn.Module):
    def __init__(self, net='densenet121', pret=True, top=False):
        super(DenseBase, self).__init__()
        self.dim = 2048
        self.top = top
        #self.no_pool = no_pool
        #if option =='vgg_bn':
        if "121" in net:
            network = models.densenet121(pretrained=pret)
        elif "161" in net:
            network = models.densenet161(pretrained=pret)
        elif "169" in net:
            network = models.densenet169(pretrained=pret)
        elif "201" in net:
            network = models.densenet201(pretrained=pret)
        if top:
            self.features = network
        else:
            self.features = nn.Sequential(*list(network.features._modules.values())[:])
        #self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):

        x = self.features(x)
        if self.top:
            return x
        else:
            x = F.relu(x, inplace=True)
            x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x


class ResNextBase(nn.Module):
    def __init__(self, net='densenet121', pret=True, top=False):
        super(ResNextBase, self).__init__()
        self.dim = 2048
        self.top = top
        if "50" in net:
            network = models.resnext50_32x4d(pretrained=True)
        elif "101" in net:
            network = models.resnext101_32x8d(pretrained=True)
        elif "wsl" in net:
            network = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        if self.top:
            self.features = network
        else:
            feature_map = list(network.children())
            feature_map.pop()
            self.features = nn.Sequential(*feature_map)

    def forward(self, x):
        x = self.features(x)
        if self.top:
            return x
        else:
            x = x.view(x.size(0), -1)
        return x


class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True):
        super(ResClassifier_MME, self).__init__()
        if norm:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        else:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.norm = norm
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if return_feat:
            return x
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
        else:
            x = self.fc(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)