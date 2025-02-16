import torch
import torch.nn as nn
from torchvision import models


class MemoBase(nn.Module):
    def __init__(self, backbone_type='resnet34', num_classes=2):
        super(MemoBase, self).__init__()
        self.backbone_type = backbone_type
        self.num_classes = num_classes
        self.specialized_blocks = nn.ModuleList()
        self.setup_backbone()

    def setup_backbone(self):
        if self.backbone_type == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            self.general_blocks = nn.Sequential(*list(backbone.children())[:-2])
            self.specialized_block = list(backbone.children())[-2]
        elif self.backbone_type.startswith('efficientnet'):
            backbone = models.efficientnet_b0(pretrained=True)
            self.general_blocks = backbone.features[:-1]
            self.specialized_block = backbone.features[-1]
        elif self.backbone_type == 'squeezenet':
            backbone = models.squeezenet1_0(pretrained=True)
            self.general_blocks = backbone.features[:-1]
            self.specialized_block = backbone.features[-1]

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self._get_fc_in_features(), self.num_classes)

    def _get_fc_in_features(self):
        if self.backbone_type == 'resnet34':
            return 512
        elif self.backbone_type.startswith('efficientnet'):
            return 1280
        elif self.backbone_type == 'squeezenet':
            return 512

    def expand_model(self, new_classes):
        # Freeze current specialized blocks
        for block in self.specialized_blocks:
            for param in block.parameters():
                param.requires_grad = False

        # Add new specialized block
        new_block = type(self.specialized_block)(*self.specialized_block.__init_args__).to(
            self.specialized_block.device)
        self.specialized_blocks.append(new_block)

        # Expand classifier
        old_fc = self.fc
        self.num_classes += new_classes
        self.fc = nn.Linear(self._get_fc_in_features(), self.num_classes).to(old_fc.device)

        # Copy weights for old classes
        with torch.no_grad():
            self.fc.weight[:old_fc.weight.shape[0]] = old_fc.weight
            self.fc.bias[:old_fc.bias.shape[0]] = old_fc.bias

    def forward(self, x):
        x = self.general_blocks(x)

        if len(self.specialized_blocks) == 0:
            x = self.specialized_block(x)
        else:
            outputs = []
            for block in self.specialized_blocks:
                outputs.append(block(x))
            x = torch.cat(outputs, dim=1)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x