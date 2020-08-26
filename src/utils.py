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


global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


# Borrowed from https://github.com/ozan-oktay/Attention-Gated-Networks
def json_file_to_pyobj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json2obj(open(filename).read())


def build_model(args, rot=False, dropout=None):

    json_options = json_file_to_pyobj(args.config)
    training_configurations = json_options.training

    modelName = training_configurations.model.lower() if not rot else 'rot' + training_configurations.model.lower()
    depth = int(training_configurations.depth)
    pretrained = True if training_configurations.pretrained == 'True' else False
    out_classes = training_configurations.out_classes
    print(out_classes)

    if modelName == 'wideresnet':
      from models.WideResNet import WideResNet
      if not pretrained:
        net = WideResNet(d=40, k=4, n_classes=out_classes, input_features=1, output_features=16, strides=[1, 1, 2, 2])
      else:
        net = WideResNet(d=40, k=4, n_classes=out_classes, input_features=3, output_features=16, strides=[1, 1, 2, 2])
      return net
    elif modelName == 'densenet':
        from models.DenseNet import DenseNet
        net = DenseNet(depth=121, growthRate=32, nClasses=out_classes)
        return net
    elif modelName == 'efficientnet':
        if depth in range(8):
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained('efficientnet-b{}'.format(depth))
            net = deepcopy(model)
            for param in net.parameters():
                param.requires_grad = True
            if not pretrained:
                net._conv_stem = nn.Conv2d(1, net._conv_stem.out_channels, kernel_size=3, stride=2, bias=False)
            net._fc = nn.Linear(model._fc.in_features, out_classes)
            if dropout is not None:
                net._dropout = torch.nn.Dropout(p=dropout)
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
            from efficientnet_pytorch.gen_odin_model import CosineSimilarity
            model._fc_nominator = CosineSimilarity(feat_dim=1280, num_centers=out_classes)
            net = deepcopy(model)
            for param in net.parameters():
                param.requires_grad = True
            return net
        else:
            raise NotImplementedError('net not implemented')
    else:
        raise NotImplementedError('net not implemented')


def build_model_with_checkpoint(modelName, model_checkpoint, device, out_classes, gen_odin_mode=2, input_features=3, rot=False):

    if 'roteb0' in modelName:
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
        model.load_state_dict(torch.load(model_checkpoint.split('.pth')[0]+'correct.pth', map_location=device), strict=False)
        os.system(f"rm {model_checkpoint.split('.pth')[0]+'correct.pth'}")
        return model
    elif 'geneb0' in modelName:
        from efficientnet_pytorch.gen_odin_model import GenOdinEfficientNet
        if 'checkpoints' not in model_checkpoint:
            model_checkpoint = os.path.join('./checkpoints', model_checkpoint)
        model = GenOdinEfficientNet.from_pretrained('efficientnet-b0', mode=gen_odin_mode)
        from efficientnet_pytorch.gen_odin_model import CosineSimilarity
        model._fc_nominator = CosineSimilarity(feat_dim=1280, num_centers=out_classes)
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
    else:
        return NotImplementedError("Model and/or checkpoint not available!")
