'Visual encoder for diffusion policy'

from typing import Callable
import torch
import torch.nn as nn
import torchvision
import torchvision.models.video as video_models


def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet

class PotentialResNetEncoder(nn.Module):
    def __init__(self, name='resnet18', output_dim=512, pretrained=False, group_norm=True):
        super().__init__()
        # 加载 ResNet 并替换最后 fc 层
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = get_resnet(name, weights=weights)

        # 修改输入层为单通道
        conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size,
                              stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)
        # 平均初始化，复制第一个通道权重或使用 kaiming init
        with torch.no_grad():
            if pretrained:
                new_conv1.weight[:] = conv1.weight[:, :1]  # copy first channel
            else:
                nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
        self.backbone.conv1 = new_conv1

        # 可选：用 GroupNorm 替换 BatchNorm
        if group_norm:
            replace_bn_with_gn(self.backbone)

        # 映射到统一的输出维度
        self.proj = nn.Linear(512, output_dim)  # resnet18/34 输出为 512

    def forward(self, x):  # x: (B, 1, 128, 128)
        x = self.backbone(x)       # -> (B, 512)
        x = self.proj(x)           # -> (B, output_dim)
        return x


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def get_resnet3d(name="r3d_18", pretrained=False):
    """
    Loads 3D ResNet model from torchvision.models.video
    Supported names: 'r3d_18', 'mc3_18', 'r2plus1d_18'
    """
    if name not in video_models.__dict__:
        raise ValueError(f"{name} not found in torchvision.models.video")

    weights = "DEFAULT" if pretrained else None
    model = video_models.__dict__[name](weights=weights)
    return model


def replace_bn3d_with_gn(model, features_per_group=16):
    """
    Replace all 3D BatchNorm layers with GroupNorm
    """
    def is_bn3d(m): return isinstance(m, nn.BatchNorm3d)

    def to_gn(m):
        return nn.GroupNorm(
            num_groups=m.num_features // features_per_group,
            num_channels=m.num_features
        )

    for name, module in model.named_modules():
        if is_bn3d(module):
            parent = model
            name_parts = name.split('.')
            for p in name_parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, name_parts[-1], to_gn(module))
    return model


class PotentialResNet3DEncoder(nn.Module):
    def __init__(self, name="r3d_18", output_dim=512, pretrained=False, group_norm=True):
        super().__init__()
        self.backbone = get_resnet3d(name, pretrained)

        # Modify input conv layer to accept 1-channel input instead of 3
        conv1 = self.backbone.stem[0]  # stem[0] is Conv3d(3, 64, kernel_size=7, ...)
        new_conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None
        )

        # Copy weights or init
        with torch.no_grad():
            if pretrained:
                new_conv1.weight[:] = conv1.weight[:, :1]  # copy channel 0
            else:
                nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
        self.backbone.stem[0] = new_conv1

        if group_norm:
            replace_bn3d_with_gn(self.backbone)

        # Replace final fc layer
        self.backbone.fc = nn.Identity()

        # Optional projection head
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x):  # x: (B, 1, 96, 96, 96)
        x = self.backbone(x)  # → (B, 512)
        x = self.proj(x)      # → (B, output_dim)
        return x
