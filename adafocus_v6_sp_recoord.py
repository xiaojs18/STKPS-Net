import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from ops.transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip
#from .resnet import resnet50
#from .mobilenet_v2 import mobilenet_v2
import torchvision.models as models
#from lib import gates
import random
class model_resnet50(nn.Module):
    def __init__(self):
        super(model_resnet50,self).__init__()

        #resnet = res2net50_26w_4s(True)
        #resnet = models.resnet50()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048,32)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self,x):
        feature = self.convnet(x)
        feature = feature.view(x.size(0), -1)
        return feature
    def get_featmap(self,x):
        for i in range(len(self.convnet)-1):
          x = self.convnet[i](x)
          if i == 7:
            feat_map = x
          
            feat_vec = self.avgpool(x)
        feat_vec=torch.flatten(feat_vec, 1)
        return feat_map,feat_vec 
    def get_featvec(self,x):
        for i in range(len(self.convnet)-1):
          x = self.convnet[i](x)
          if i == 7:
            feat_vec = self.avgpool(x)
        feat_vec=torch.flatten(feat_vec, 1)
        return feat_vec  
def get_patches_frame(input_frames, actions, patch_size, image_size):
    input_frames = input_frames.view(-1, 3, image_size, image_size)  # [NT,C,H,W]
    theta = torch.zeros(input_frames.size(0), 2, 3).cuda()
    #patch_coordinate = (actions * (image_size - patch_size))
    #coord_map_x=torch.tensor([[[ [-48,-32,-16,0,16,32,48],
    #         [-48,-32,-16,0,16,32,48],
    #          [-48,-32,-16,0,16,32,48],
    #          [-48,-32,-16,0,16,32,48],
    #          [-48,-32,-16,0,16,32,48],
    #          [-48,-32,-16,0,16,32,48],
    #          [-48,-32,-16,0,16,32,48],
    #            ]]]).cuda()
    coord_map_x=torch.tensor([[[ [0,16,32,48,64,80,96],
              [0,16,32,48,64,80,96],
              [0,16,32,48,64,80,96],
              [0,16,32,48,64,80,96],
              [0,16,32,48,64,80,96],
              [0,16,32,48,64,80,96],
              [0,16,32,48,64,80,96],
                ]]]).cuda()
    coord_map_x=coord_map_x.repeat(actions.size(0),1,1,1).float()
    #coord_map_y=torch.tensor([[[ [48,48,48,48,48,48],
    #          [32,32,32,32,32,32],
    #          [16,16,16,16,16,16],
    #          [-16,-16,-16,-16,-16,-16],
    #         [-32,-32,-32,-32,-32,-32],
    #         [-48,-48,-48,-48,-48,-48],
    #            ]]]).cuda()
    #coord_map_y=torch.tensor([[[ [-48,-48,-48,-48,-48,-48,-48],
    #          [-32,-32,-32,-32,-32,-32,-32],
    #          [-16,-16,-16,-16,-16,-16,-16],
    #          [0,0,0,0,0,0,0],
    #          [16,16,16,16,16,16,16],
    #         [32,32,32,32,32,32,32],
    #         [48,48,48,48,48,48,48],
    #            ]]]).cuda()
    coord_map_y=torch.tensor([[[ [0,0,0,0,0,0,0],
              [16,16,16,16,16,16,16],
              [32,32,32,32,32,32,32],
              [48,48,48,48,48,48,48],
              [64,64,64,64,64,64,64],
             [80,80,80,80,80,80,80],
             [96,96,96,96,96,96,96],
                ]]]).cuda()
    coord_map_y=coord_map_y.repeat(actions.size(0),1,1,1).float() 
    #coord_x=(coord_map_x*actions).sum(dim=(2,3))+image_size//2
    #coord_y=(coord_map_y*actions).sum(dim=(2,3))+image_size//2
    #coord_x=(coord_map_x*actions).sum(dim=(2,3))+patch_size//2
    #coord_y=(coord_map_y*actions).sum(dim=(2,3))+patch_size//2
    coord_x=((coord_map_x*actions).sum(dim=(2,3)))/10
    coord_y=((coord_map_y*actions).sum(dim=(2,3)))/10
    #coord_x=((coord_map_x*actions).sum(dim=(2,3)))/10-image_size//2
    #coord_y=((coord_map_y*actions).sum(dim=(2,3)))/10-image_size//2
    #print(coord_x)
    #print(coord_y)
    coord_x=torch.clamp(coord_x,patch_size//2,image_size-patch_size//2-1)
    coord_y=torch.clamp(coord_y,patch_size//2,image_size-patch_size//2-1)
    #x1, x2, y1, y2 = patch_coordinate[:, 1], patch_coordinate[:, 1] + patch_size, \
                    # patch_coordinate[:, 0], patch_coordinate[:, 0] + patch_size
    x1, x2, y1, y2 = coord_x[:,0]-patch_size//2, coord_x[:,0] + patch_size//2, \
                     coord_y[:,0]-patch_size//2, coord_y[:,0] + patch_size//2
    
    theta[:, 0, 0], theta[:, 1, 1] = patch_size / image_size, patch_size / image_size
    theta[:, 0, 2], theta[:, 1, 2] = -1 + (x1 + x2) / image_size, -1 + (y1 + y2) / image_size

    grid = F.affine_grid(theta.float(), torch.Size((input_frames.size(0), 3, patch_size, patch_size)))
    patches = F.grid_sample(input_frames, grid)  # [NT,C,H1,W1]
    return patches

class arg:
    def __init__(self):
        self.glance_arch='res50'
        self.workers=8 
        self.num_segments=8 
        self.dropout=0.2  
        self.fc_dropout=0.2
        self.batch_size=32
        self.patch_size=128 
        self.global_lr_ratio=0.5 
        self.stn_lr_ratio=0.1 
        self.classifier_lr_ratio=5.0
        self.hidden_dim=1024 
        self.stn_hidden_dim=2048
        self.num_classes=32
        self.dataset='minikinetics'
        self.input_size=224 
        self.glance_ckpt_path=''
        
class AdaFocus(nn.Module):
    def __init__(self,num_classes,modality):
        super(AdaFocus, self).__init__()
        
        args=arg()
        
        self.stn_lr_ratio=args.stn_lr_ratio
        self.global_lr_ratio=args.global_lr_ratio
        self.num_segments = args.num_segments
        self.num_classes = num_classes
        if args.dataset == 'fcvid':
            assert args.num_classes == 239
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.glance_arch = args.glance_arch
        if args.glance_arch == 'mbv2':
            print('Global CNN Backbone: mobilenet_v2')
            self.global_CNN = mobilenet_v2(pretrained=True)
            self.global_feature_dim = self.global_CNN.last_channel
            self.stn_state_dim = self.global_feature_dim * 7 * 7
        elif args.glance_arch == 'res50':
            print('Global CNN Backbone: resnet50 (glance size 96*96)')
            self.global_CNN = model_resnet50()
            
            self.global_feature_dim = self.global_CNN.fc.in_features
            #self.stn_state_dim = self.global_feature_dim * 3 * 3
            self.glance_size = 96
        else:
            raise NotImplementedError

        #self.global_CNN_fc = nn.Sequential(
        #    nn.Dropout(args.fc_dropout),
        #    nn.Linear(self.global_feature_dim, self.num_classes),
        #)
        
        self.local_CNN = model_resnet50()
        self.local_feature_dim = self.local_CNN.fc.in_features
        #if(modality=='image'):
        #    self.global_CNN.load_state_dict(torch.load('./model/hmdb_rgb_submodel.pkl'))
        #    self.local_CNN.load_state_dict(torch.load('./model/hmdb_rgb_submodel.pkl'))
        #    print('submodel_rgb loaded from {}'.format('./model/hmdb_rgb_submodel.pkl'))
        #if(modality=='depth'):
        #   self.global_CNN.load_state_dict(torch.load('./model/hmdb_depth_submodel.pkl'))
        #    self.local_CNN.load_state_dict(torch.load('./model/hmdb_depth_submodel.pkl'))
        #    print('submodel_depth loaded from {}'.format('./model/hmdb_depth_submodel.pkl'))
        if(modality=='nouse'):
            pass
        #self.local_CNN_fc = nn.Sequential(
           # nn.Dropout(args.fc_dropout),
           # nn.Linear(self.local_feature_dim, self.num_classes),
        #)

        self.stn_feature_dim = self.global_feature_dim
        #self.policy_stn = PolicySTN(
         #   stn_state_dim=self.stn_state_dim,
         #   stn_feature_dim=self.global_feature_dim,
          #  num_segments=args.num_segments,
          #  hidden_dim=args.stn_hidden_dim,
        #)
        #self.policy_stn=PolicySTN(
                #args.stn_hidden_dim, 2, 1, 0
           #)#kernel_size=2 stride=1 padding=0 
        self.cat_feature_dim = self.global_feature_dim + self.local_feature_dim
        #self.cat_CNN_fc = nn.Sequential(
        #    nn.Dropout(args.fc_dropout),
        #    nn.Linear(self.cat_feature_dim, self.num_classes),
        #)

        #self.cat_feature_dim=self.num_classes+self.num_classes
        #print(self.cat_feature_dim)
        #self.classifier = PoolingClassifier(
        #    input_dim=self.cat_feature_dim,
         #   num_segments=self.num_segments,
        #   num_classes=args.num_classes,
        #    dropout=args.dropout
        #)

    def forward(self, images,mode):
        if self.glance_arch == 'res50':
            
            images = images.view(-1, 3, self.input_size, self.input_size)
            
            global_feat_map, global_feat = self.global_CNN.get_featmap(images)#global_feat_map -1*2048*7*7 global_feat -1*2048*1*1
        nt, c, h, w = global_feat_map.size()
        n=nt//self.num_segments
        #global_logits = self.global_CNN_fc(global_feat)
        global_feat_map=global_feat_map.mean(1, keepdim=True)
        
        #global_feat_map=global_feat_map.view(-1, 1, 7*7)#this is very important
        #global_feat_map=F.softmax(global_feat_map,dim=2)
        #global_feat_map=global_feat_map.view(-1, 1, 7,7)
        
        if(mode=='train'):
        #global_feat_map=global_feat_map.view(n*self.num_segments, 1, 7*7)
        #global_feat_map=F.softmax(global_feat_map,dim=2)
        #global_feat_map=global_feat_map.view(n*self.num_segments, 1, 7,7)
        
            #actions_1 = self.policy_stn(global_feat_map.detach())
            #actions_2 = torch.rand_like(actions_1)
            a=random.random()
            if (a>0.5):
             global_feat_map_1 = torch.rand_like(global_feat_map)
             #global_feat_map_1=global_feat_map_1.view(-1, 1, 7*7)#this is very important
             #global_feat_map_1=F.softmax(global_feat_map_1,dim=2)
             #global_feat_map_1=global_feat_map_1.view(-1, 1, 7,7)
             patches = get_patches_frame(images, global_feat_map_1.detach(), self.patch_size, self.input_size)
            #res = []
            #for actions in [actions_1, actions_2]:
            else:
             patches = get_patches_frame(images, global_feat_map.detach(), self.patch_size, self.input_size)
            local_feat = self.local_CNN.get_featvec(patches)
            #local_logits = self.local_CNN_fc(local_feat)
            cat_feat = torch.cat([global_feat, local_feat], dim=-1)#xiugaile
            #cat_feat = cat_feat.view(-1, self.num_segments, self.cat_feature_dim)
            #cat_logits, cat_pred = self.classifier(cat_feat)#xiugaishuchu
            #res.append((cat_feat,  global_logits, local_logits))
            return cat_feat,global_feat,local_feat
        else:
            #actions = self.policy_stn(global_feat_map)
            patches = get_patches_frame(images, global_feat_map.detach(), self.patch_size, self.input_size)
            local_feat = self.local_CNN.get_featvec(patches)
            #local_logits = self.local_CNN_fc(local_feat)
            cat_feat = torch.cat([global_feat, local_feat], dim=-1)
            #cat_feat = cat_feat.view(-1, self.num_segments, self.cat_feature_dim)
            #cat_logits, cat_pred = self.classifier(cat_feat)
            return cat_feat,global_feat,local_feat
    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    @property
    def crop_size(self):
        return self.input_size

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose(
                [GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]), GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])

    def get_optim_policies(self):
        return [{'params': self.local_CNN.parameters(), 'lr_mult': 1, 'decay_mult': 1, 'name': "local_CNN"}] \
               + [{'params': self.global_CNN.parameters(), 'lr_mult': 1, 'decay_mult': 1,
                   'name': "global_CNN"}]  \

class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1)), dim=1)
        return x.max(dim=1)[0]


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, num_neurons=4096):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = [num_neurons]
        layers = []
        dim_input = input_dim
        for dim_output in self.num_neurons:
            layers.append(nn.Linear(dim_input, dim_output))
            layers.append(nn.BatchNorm1d(dim_output))
            layers.append(nn.ReLU())
            dim_input = dim_output
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class PoolingClassifier(nn.Module):
    def __init__(self, input_dim, num_segments, num_classes, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_pooling = MaxPooling()
        self.mlp = MultiLayerPerceptron(input_dim)
        self.num_segments = 16
        self.classifiers = nn.ModuleList()
        for m in range(self.num_segments):
            self.classifiers.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(4096, self.num_classes)
            ))

    def forward(self, x):
        _b = x.size(0)
        x = x.view(-1, self.input_dim)
        z = self.mlp(x).view(_b, self.num_segments, -1)
        logits = torch.zeros(_b, self.num_segments, self.num_classes).cuda()
        cur_z = z[:, 0]
        for frame_idx in range(0, self.num_segments):
            if frame_idx > 0:
                cur_z = self.max_pooling(z[:, frame_idx], cur_z)
            logits[:, frame_idx] = self.classifiers[frame_idx](cur_z)
        last_out = logits[:, -1, :].reshape(_b, -1)
        logits = logits.view(_b * self.num_segments, -1)
        return logits, last_out


class PolicySTN(nn.Module):
    def __init__(self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        ):
        super(PolicySTN, self).__init__()
        self.gating_layers = nn.Conv2d(
                in_channels, 1, kernel_size=kernel_size, stride=stride, padding=padding
            )
        
        #self.sigmoid=nn.Sigmoid()

        #self.softmax=F.softmax()

    def forward(self, features):
        feature = self.gating_layers(features)  # [NT, H]
        feature=F.softmax(feature,dim=-1)
        return feature
      #self.sigmoid=nn.Sigmoid()

        #self.softmax=F.softmax()

    def forward(self, features):
        feature = self.gating_layers(features)  # [NT, H]
        feature=F.softmax(feature,dim=-1)
        return feature
