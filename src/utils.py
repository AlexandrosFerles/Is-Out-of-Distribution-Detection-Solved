from torch import nn as nn
import torch
import torchvision
import collections
from collections import OrderedDict
import os
import json
from copy import deepcopy
import random
import ipdb


# global_seed = 1
# torch.backends.cudnn.deterministic = True
# random.seed(global_seed)
# torch.manual_seed(global_seed)
# torch.cuda.manual_seed(global_seed)


# Borrowed from https://github.com/ozan-oktay/Attention-Gated-Networks
def json_file_to_pyobj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json2obj(open(filename).read())


def build_model(args):

    json_options = json_file_to_pyobj(args.config)
    training_configurations = json_options.training

    modelName = training_configurations.model.lower()
    depth = int(training_configurations.depth)
    pretrained = True if training_configurations.pretrained == 'True' else False
    out_classes = training_configurations.out_classes

    if modelName == 'wideresnet':
      from models.WideResNet import WideResNet
      net = WideResNet(d=40, k=4, n_classes=out_classes, input_features=3, output_features=16, strides=[1, 1, 2, 2])
      return net
    elif modelName == 'resnet':
        if depth == 50:
            pretrained = torchvision.models.resnet50(pretrained=pretrained)
        elif depth == 101:
            pretrained = torchvision.models.resnet101(pretrained=pretrained)
        elif depth == 152:
            pretrained = torchvision.models.resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError('ResNet depth {} not implemented'.format(depth))
        net = deepcopy(pretrained)
        for param in net.parameters():
            param.requires_grad = True
        net.fc = nn.Linear(net.fc.in_features, out_classes)
        return net 
    elif modelName == 'senet':
        if depth == 154:
            from models.SENets import senet154
            pretrained = senet154(num_classes=1000, pretrained=pretrained)
            net = deepcopy(pretrained)
            for param in net.parameters():
                param.requires_grad = True
            net.last_linear = nn.Linear(net.last_linear.in_features, out_classes)
            return net
        else:
            raise NotImplementedError('SENet depth {} not implemented'.format(depth))
    elif modelName == 'seresnext':
        if depth == 50:
            from pretrainedmodels import se_resnext50_32x4d
            pretrained = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
            net = deepcopy(pretrained)
            for param in net.parameters():
                param.requires_grad = True
            net.last_linear = nn.Linear(net.last_linear.in_features, out_classes)
            return net
        elif depth == 101:
            from pretrainedmodels import se_resnext101_32x4d
            pretrained = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
            net = deepcopy(pretrained)
            for param in net.parameters():
                param.requires_grad = True
            net.last_linear = nn.Linear(net.last_linear.in_features, out_classes)
            return net
        else:
            raise NotImplementedError('SE-ResNext depth {} not implemented'.format(depth))
    elif modelName == 'densenet':
        if depth == 121:
            from pretrainedmodels import densenet121
            pretrained = densenet121(num_classes=1000, pretrained='imagenet')
            net = deepcopy(pretrained)
            for param in net.parameters():
                param.requires_grad = True
            net.last_linear = nn.Linear(net.last_linear.in_features, out_classes)
            return net
        elif depth == 169:
            from pretrainedmodels import densenet169
            pretrained = densenet169(num_classes=1000, pretrained='imagenet')
            net = deepcopy(pretrained)
            for param in net.parameters():
                param.requires_grad = True
            net.last_linear = nn.Linear(net.last_linear.in_features, out_classes)
            return net
        elif depth == 201:
            from pretrainedmodels import densenet201
            pretrained = densenet201(num_classes=1000, pretrained='imagenet')
            net = deepcopy(pretrained)
            for param in net.parameters():
                param.requires_grad = True
            net.last_linear = nn.Linear(net.last_linear.in_features, out_classes)
            return net
        else:
            raise NotImplementedError('net not implemented')
    elif modelName == 'efficientnet':
        if depth in range(8):
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b{}'.format(depth))
            net = deepcopy(model)
            for param in net.parameters():
                param.requires_grad = True
            ipdb.set_trace()
            if not pretrained:
                net._conv_stem = nn.Conv2d(1, net._conv_stem.out_channels, kernel_size=3, stride=2, bias=False)
            net._fc = nn.Linear(model._fc.in_features, out_classes)
            return net
        else:
            raise NotImplementedError('net not implemented')
    elif modelName == 'rotefficientnet':
        if depth in range(8):
            from efficientnet_pytorch.rot_model import RotEfficientNet
            model = RotEfficientNet.from_pretrained('efficientnet-b{}'.format(depth))
            net = deepcopy(model)
            for param in net.parameters():
                param.requires_grad = True
            net._fc = nn.Linear(model._fc.in_features, out_classes)
            return net
        else:
            raise NotImplementedError('net not implemented')
    elif modelName == 'genodinefficientnet':
        gen_odin_mode = training_configurations.gen_odin_mode
        if depth in range(8):
            from efficientnet_pytorch.gen_odin_model import GenOdinEfficientNet
            model = GenOdinEfficientNet.from_pretrained('efficientnet-b{}'.format(depth), mode=gen_odin_mode)
            net = deepcopy(model)
            for param in net.parameters():
                param.requires_grad = True
            return net
        else:
            raise NotImplementedError('net not implemented')
    else:
        raise NotImplementedError('net not implemented')


def build_model_with_checkpoint(modelName, model_checkpoint, device, out_classes, gen_odin_mode=0):

    if 'wide' in modelName:
        from models.WideResNet import WideResNet
        strides = [1, 1, 2, 2]
        net = WideResNet(d=40, k=2, n_classes=10, input_features=3, output_features=16, strides=strides, rot=True)
        if 'checkpoints' not in model_checkpoint:
            model_checkpoint = os.path.join('./checkpoints', model_checkpoint)
        state_dict = torch.load(model_checkpoint, map_location=device)
        net.load_state_dict(state_dict)
        net = net.to(device)
        return net
    elif 'genDense' in modelName:
        from models.DenseNet import DenseNet
        print(gen_odin_mode)
        net = DenseNet(mode=gen_odin_mode)
        if 'checkpoints' not in model_checkpoint:
            model_checkpoint = os.path.join('./checkpoints', model_checkpoint)
        state_dict = torch.load(model_checkpoint, map_location=device)
        net.load_state_dict(state_dict)
        net = net.to(device)
        return net
    elif 'senet154' in modelName:
        from models.SENets import senet154
        net = senet154(num_classes=1000)
        net.last_linear = nn.Linear(net.last_linear.in_features, out_classes)
        net = net.to(device)
        print('Loading model....')
        if 'checkpoints' not in model_checkpoint:
            model_checkpoint = os.path.join('./checkpoints', model_checkpoint)
        state_dict = torch.load(os.path.join(model_checkpoint))
        new_state_dict = collections.OrderedDict()
        for key, value in state_dict.items():
            new_key = key.split('module.')[1]
            new_state_dict[new_key] = value
        torch.save(new_state_dict, os.path.join(model_checkpoint).split('.pth')[0]+'correct.pth')
        net.load_state_dict(torch.load(os.path.join(model_checkpoint).split('.pth')[0]+'correct.pth', map_location=device))
        os.system(f"rm {os.path.join(model_checkpoint).split('.pth')[0]+'correct.pth'}")
    elif 'seresnext50' in modelName:
        from pretrainedmodels import se_resnext50_32x4d
        net = se_resnext50_32x4d(num_classes=1000)
        net.last_linear = nn.Linear(net.last_linear.in_features, out_classes)
        if 'checkpoints' not in model_checkpoint:
            model_checkpoint = os.path.join('./checkpoints', model_checkpoint)
        state_dict = torch.load(model_checkpoint, map_location=device)
        net.load_state_dict(state_dict)
        net = net.to(device)
        return net
    elif 'seresnext101' in modelName:
        from pretrainedmodels import se_resnext101_32x4d
        net = se_resnext101_32x4d(num_classes=1000)
        net.last_linear = nn.Linear(net.last_linear.in_features, out_classes)
        if 'checkpoints' not in model_checkpoint:
            model_checkpoint = os.path.join('./checkpoints', model_checkpoint)
        state_dict = torch.load(model_checkpoint, map_location=device)
        net.load_state_dict(state_dict)
        net = net.to(device)
        return net
    elif 'densenet121' in modelName:
        from pretrainedmodels import densenet121
        net = densenet121(num_classes=1000)
        net.last_linear = nn.Linear(net.last_linear.in_features, out_classes)
        if 'checkpoints' not in model_checkpoint:
            model_checkpoint = os.path.join('./checkpoints', model_checkpoint)
        state_dict = torch.load(model_checkpoint, map_location=device)
        net.load_state_dict(state_dict)
        net = net.to(device)
        return net
    elif 'densenet169' in modelName:
        from pretrainedmodels import densenet169
        net = densenet169(num_classes=1000)
        net.last_linear = nn.Linear(net.last_linear.in_features, out_classes)
        model_checkpoint = os.path.join(model_checkpoint)
        state_dict = torch.load(model_checkpoint, map_location=device)
        net.load_state_dict(state_dict)
        net = net.to(device)
        return net
    elif 'densenet201' in modelName:
        from pretrainedmodels import densenet201
        net = densenet201(num_classes=1000)
        net.last_linear = nn.Linear(net.last_linear.in_features, out_classes)
        if 'checkpoints' not in model_checkpoint:
            model_checkpoint = os.path.join('./checkpoints', model_checkpoint)
        state_dict = torch.load(model_checkpoint, map_location=device)
        net.load_state_dict(state_dict)
        net = net.to(device)
        return net
    elif 'roteb0' in modelName:
        from efficientnet_pytorch.rot_model import RotEfficientNet
        model = RotEfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Linear(model._fc.in_features, out_classes)
        model = model.to(device)
        if 'checkpoints' not in model_checkpoint:
            model_checkpoint = os.path.join('./checkpoints', model_checkpoint)
        state_dict = torch.load(model_checkpoint, map_location=device)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.split('module.')[1]
            new_state_dict[new_key] = value
        torch.save(new_state_dict, model_checkpoint.split('.pth')[0]+'correct.pth')
        model.load_state_dict(torch.load(model_checkpoint.split('.pth')[0]+'correct.pth', map_location=device))
        os.system(f"rm {model_checkpoint.split('.pth')[0]+'correct.pth'}")
        return model
    elif 'geneb0' in modelName:
        from efficientnet_pytorch.gen_odin_model import GenOdinEfficientNet
        if 'checkpoints' not in model_checkpoint:
            model_checkpoint = os.path.join('./checkpoints', model_checkpoint)
        model = GenOdinEfficientNet.from_pretrained('efficientnet-b0', mode=gen_odin_mode)
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        model = model.to(device)
        return model
    elif 'eb0' in modelName:
        from efficientnet_pytorch import EfficientNet
        net = EfficientNet.from_name('efficientnet-b0')
        net._fc = nn.Linear(net._fc.in_features, out_classes)
        if 'checkpoints' not in model_checkpoint:
            model_checkpoint = os.path.join('./checkpoints', model_checkpoint)
        state_dict = torch.load(model_checkpoint, map_location=device)
        net.load_state_dict(state_dict)
        net = net.to(device)
        return net
    elif 'geneb6' in modelName:
        from efficientnet_pytorch.gen_odin_model import GenOdinEfficientNet
        model = GenOdinEfficientNet.from_name('efficientnet-b6')
        net = deepcopy(model)
        net._fc_nominator = nn.Linear(model._fc_nominator.in_features, out_classes)
        net._fc_denominator = nn.Linear(model._fc_denominator.in_features, out_classes)
        net._denominator_batch_norm = nn.BatchNorm1d(out_classes)
        net = net.to(device)
        print('Loading model....')
        state_dict = torch.load(os.path.join(model_checkpoint))
        new_state_dict = collections.OrderedDict()
        for key, value in state_dict.items():
            new_key = key.split('module.')[1]
            new_state_dict[new_key] = value
        torch.save(new_state_dict, os.path.join(model_checkpoint).split('.pth')[0]+'correct.pth')
        net.load_state_dict(torch.load(os.path.join(model_checkpoint).split('.pth')[0]+'correct.pth', map_location=device))
        os.system(f"rm {os.path.join(model_checkpoint).split('.pth')[0]+'correct.pth'}")
        return net
    elif 'eb6' in modelName:
        ipdb.set_trace()
        from efficientnet_pytorch import EfficientNet
        net = EfficientNet.from_name('efficientnet-b6')
        net._fc = nn.Linear(net._fc.in_features, out_classes)
        net = net.to(device)
        print('Loading model....')
        state_dict = torch.load(os.path.join(model_checkpoint))
        new_state_dict = collections.OrderedDict()
        for key, value in state_dict.items():
            new_key = key.split('module.')[1]
            new_state_dict[new_key] = value
        torch.save(new_state_dict, os.path.join(model_checkpoint).split('.pth')[0]+'correct.pth')
        net.load_state_dict(torch.load(os.path.join(model_checkpoint).split('.pth')[0]+'correct.pth', map_location=device))
        os.system(f"rm {os.path.join(model_checkpoint).split('.pth')[0]+'correct.pth'}")
        return net
    else:
        return NotImplementedError("Model and/or checkpoint not available!")
