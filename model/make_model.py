import torch
import torch.nn as nn
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, gem_pool=cfg.MODEL.GEM_POOLING, stem_conv=cfg.MODEL.STEM_CONV,
                                                       moe_layer_idx=cfg.MOE.BLOCKS, num_experts=cfg.MOE.NUM_EXPERTS, top_k=cfg.MOE.TOP_K)
        self.in_planes = self.base.in_planes
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
       
        if self.reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.in_planes = cfg.MODEL.FEAT_DIM

        self.classifier_shared = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_shared.apply(weights_init_classifier)

        self.classifier_day = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_day.apply(weights_init_classifier)

        self.classifier_night = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_night.apply(weights_init_classifier)

        self.bottleneck_shared = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_shared.bias.requires_grad_(False)
        self.bottleneck_shared.apply(weights_init_kaiming)

        self.bottleneck_domain = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_domain.bias.requires_grad_(False)
        self.bottleneck_domain.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.load_param(model_path)

    def forward(self, x, label=None, cam_label= None, view_label=None, domain_label=None):
        x_shared, x_domain, domain_logit = self.base(x, cam_label=cam_label, view_label=view_label, domain_label=domain_label)

        if self.reduce_feat_dim:
            x_shared = self.fcneck(x_shared)
            x_domain = self.fcneck(x_domain)


        feat_shared = self.bottleneck_shared(x_shared)
        feat_domain = self.bottleneck_domain(x_domain)

        feat_shared_cls = self.dropout(feat_shared)
        feat_domain_cls = self.dropout(feat_domain)

        if self.training:
            cls_score_shared = self.classifier_shared(feat_shared_cls)
            cls_score_day = self.classifier_day(feat_domain_cls)
            cls_score_night = self.classifier_night(feat_domain_cls)

            return {
                'shared': cls_score_shared,
                'day': cls_score_day,
                'night': cls_score_night
            }, feat_shared, domain_logit  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat_shared
            else:
                # print("Test with feature before BN")
                return x_shared

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))



__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
}

def make_model(cfg, num_class, camera_num, view_num):

    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    print('===========building transformer===========')

    return model
