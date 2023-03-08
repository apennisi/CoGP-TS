import torch
from torch import nn

from modules.conv import conv, conv_dw, conv_dw_no_bn
from models.mobilenetv3 import MobileNetV3
import collections
from utils.config import config

class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs, num_backbone_feat=512):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, num_backbone_feat, kernel_size=1, padding=0, bn=False),
            conv(num_backbone_feat, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, num_backbone_feat, kernel_size=1, padding=0, bn=False),
            conv(num_backbone_feat, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


def load_from_mobilenetv3(net, checkpoint):
    print(checkpoint.keys())
    # source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    to_exclude = ["classifier.1.weight", "classifier.1.bias", 'features.14.weight', 'features.14.bias', 'classifier.1.weight', 'classifier.1.bias']
    # , "features.12.0.weight", "features.12.1.weight", "features.12.0.bias", "features.12.1.bias",
    # 'features.12.1.running_mean', 'features.12.1.running_var', 'features.12.1.num_batches_tracked', 'features.14.weight', 'features.14.bias', 'classifier.1.weight', 'classifier.1.bias']
    for target_key, target_value in checkpoint.items():
        if target_key in to_exclude:
          continue
        k = target_key
        if k.find('model') != -1:
            k = k.replace('model', 'module.model')
        if k in checkpoint and checkpoint[k].size() == target_state[target_key].size():
            new_target_state[target_key] = checkpoint[k]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)


class PoseEstimationWithMobileNetV3(nn.Module):
  def __init__(self, num_refinement_stages=config['num_refinement_steps'], num_channels=128, num_heatmaps=config['keypoint_number']+1, num_pafs=len(config['body_parts_paf_ids'])*2, pretrained='mobilenetv3_small_67.4.pth.tar'):
    super().__init__()
    self.model = MobileNetV3()
    if pretrained:
        checkpoint = torch.load(pretrained)
        load_from_mobilenetv3(self.model, checkpoint)
        # state_dict = torch.load(pretrained)
        # self.model.load_state_dict(state_dict, strict=True)
    
    self.cpm = Cpm(576, num_channels)
    self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs, num_backbone_feat=576)
    self.refinement_stages = nn.ModuleList()
    for _ in range(num_refinement_stages):
        self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                      num_heatmaps, num_pafs))

  def forward(self, x):
      # print("input shape", x.shape)
      backbone_features = self.model(x)
      # print(backbone_features.shape)
      backbone_features = self.cpm(backbone_features)
      # print(backbone_features.shape)

      stages_output = self.initial_stage(backbone_features)
      for refinement_stage in self.refinement_stages:
          stages_output.extend(
              refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))
      # for stage in stages_output:
        # print(stage.shape)
      return stages_output



class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=6, num_pafs=8):
        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for _ in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))
        
        return stages_output
