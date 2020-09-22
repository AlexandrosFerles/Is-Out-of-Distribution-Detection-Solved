import torch
from torch import nn
from torch.nn import functional as F
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)
import math

class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

class CentroidEuclideanDist(nn.Module):
    def __init__(self, feat_dim, num_centers):
        super(CentroidEuclideanDist, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, feat_dim))
        torch.nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))  # The init is the same as nn.Linear

    def forward(self, feat):
        diff = feat.unsqueeze(dim=1) - self.centers.unsqueeze(dim=0)  # Broadcasting operation
        diff.pow_(2)
        dist = diff.sum(dim=-1)
        return dist


class CosineSimilarity(nn.Module):
    def __init__(self, feat_dim, num_centers):
        super(CosineSimilarity, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, feat_dim))
        torch.nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))  # The init is the same as nn.Linear

    def forward(self, feat):
        feat_normalized = F.normalize(feat)
        center_normalized = F.normalize(self.centers)

        return torch.mm(feat_normalized, center_normalized.t())


class GenOdinEfficientNet(nn.Module):

    def __init__(self, blocks_args=None, global_params=None, mode=0, out_classes=10):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self.mode = mode
        self._fc_denominator = nn.Linear(out_channels, 1)
        self._denominator_batch_norm = nn.BatchNorm1d(1)
        self._denominator_sigmoid = torch.nn.Sigmoid()
        if self.mode == 0:
            self._fc_nominator = nn.Linear(out_channels, out_classes)
            torch.nn.init.kaiming_uniform_(self._fc_nominator.weight, a=math.sqrt(5))
        elif self.mode == 1:
            self._fc_nominator = CentroidEuclideanDist(out_channels, out_classes)
        else:
            self._fc_nominator = CosineSimilarity(out_channels, out_classes)

        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    def extract_features(self, inputs, mode='final'):
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        if mode !='final':
            intermediate_features = []

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if mode !='final':
                intermediate_features.append(x)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        if mode == 'final':
            return x
        else:
            return x, intermediate_features

    def forward(self, inputs, mode=0):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        bs = inputs.size(0)

        x = self.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        g = self._fc_denominator(x)
        g = self._denominator_batch_norm(g)
        g = self._denominator_sigmoid(g)
        h = self._fc_nominator(x)

        if self.mode == 1:
            x = - h / g
        else:
            x = h / g
        x = self._dropout(x)

        return x, h, g

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, in_channels = 3, mode=0):
        # model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        blocks_args, global_params = get_model_params(model_name, override_params={'num_classes': num_classes})
        model = GenOdinEfficientNet(blocks_args=blocks_args, global_params=global_params, mode=mode)
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), rot=True)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, mode=0):
        # model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        blocks_args, global_params = get_model_params(model_name, override_params={'num_classes': num_classes})
        model = GenOdinEfficientNet(blocks_args=blocks_args, global_params=global_params, mode=mode)
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), rot=True)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet-b'+str(i) for i in range(num_models)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))