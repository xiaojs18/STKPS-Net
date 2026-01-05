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
def get_patches_frame(input_frames, actions,patch_size, image_size):
    input_frames = input_frames.view(-1, 3, image_size, image_size)  # [NT,C,H,W]
    theta = torch.zeros(input_frames.size(0), 2, 3).cuda()
    #patch_coordinate = (actions * (image_size - patch_size))
    #coord_map_x=torch.tensor([[[ [-48,-32,-16,0,16,32,48],
    #          [-48,-32,-16,0,16,32,48],
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
    #coord_map_x=torch.tensor([[[ [0,16,32,48,64,80,96]]]]).cuda()
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
    #coord_map_y=torch.tensor([[[ [0],
    #          [16],
    #          [32],
    #          [48],
    #          [64],
    #         [80],
    #         [96],
    #            ]]]).cuda()
    coord_map_y=coord_map_y.repeat(actions.size(0),1,1,1).float() 
    #coord_x=(coord_map_x*actions).sum(dim=(2,3))+image_size//2
    #coord_y=(coord_map_y*actions).sum(dim=(2,3))+image_size//2
    #coord_x=((coord_map_x*actions).sum(dim=(2,3)))/10+patch_size//2
    #coord_y=((coord_map_y*actions).sum(dim=(2,3)))/10+patch_size//2
    coord_x=((coord_map_x*actions).sum(dim=(2,3)))/10
    coord_y=((coord_map_y*actions).sum(dim=(2,3)))/10
    #coord_x=(coord_map_x*actionsx).sum(dim=(2,3))+image_size//2
    #coord_y=(coord_map_y*actionsy).sum(dim=(2,3))+image_size//2
    #print(coord_x,'xxxxxxxxxxxxxxxxxxxxxxxxxx')
    #print(coord_y,'yyyyyyyyyyyyyyyyyyy')
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
        self.sigmoid=nn.Sigmoid()
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
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), 
                                    stride=(1, 1 ,1), bias=False, padding=(1, 1, 1))
        #self.action_p1_squeeze1 = nn.Conv2d(2048*self.num_segments, self.num_segments, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p1_squeeze1 = nn.Conv2d(self.num_segments, self.num_segments, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        #nn.init.xavier_uniform_(self.action_p1_conv1.weight)
        #self.action_p1_conv1.bias.data.fill_(0)
        self.pad = (0,0,0,0,0,0,0,1)
        
        self.action_p2_squeeze = nn.Conv2d(2048, 128, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p2_conv1 = nn.Conv1d(128, 128, kernel_size=3, stride=1, bias=False, padding=1, 
                                       groups=1)
        self.action_p2_expand = nn.Conv2d(128, 2048, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        
        self.action_p2_squeeze_global = nn.Conv2d(2048, 128, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p2_conv1_global = nn.Conv1d(128, 128, kernel_size=3, stride=1, bias=False, padding=1, 
                                       groups=1)
        self.action_p2_expand_global = nn.Conv2d(128, 2048, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        
        
        self.action_p3_squeeze = nn.Conv2d(2048, 1, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p3_bn1 = nn.BatchNorm2d(1)
        self.action_p3_conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3), 
                                    stride=(1 ,1), bias=False, padding=(1, 1), groups=1)
                                    
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, images,mode):
        if self.glance_arch == 'res50':
            
            images = images.view(-1, 3, self.input_size, self.input_size)
            
            global_feat_map, global_feat = self.global_CNN.get_featmap(images)#global_feat_map -1*2048*7*7 global_feat -1*2048*1*1
        nt, c, h, w = global_feat_map.size()
        n=nt//self.num_segments
        #global_logits = self.global_CNN_fc(global_feat)
        
        
        #global_feat_map=global_feat_map.detach()
        # 添加这段代码：确保帧数是num_segments的倍数,修改
        # valid_nt = n * self.num_segments
        # if nt != valid_nt:
        #     global_feat_map = global_feat_map[:valid_nt]
        #     global_feat = global_feat[:valid_nt]
        #     nt = valid_nt
        #     images = images[:valid_nt]  # 关键：同时截断images，修改

        # 修改keepdim=False
        global_feat_map_fow=global_feat_map.detach().mean(1, keepdim=True)

        global_feat_map_fow = global_feat_map_fow.view(n, self.num_segments,h, w)#zhuyibunengfenka
        #global_feat_map_fow = global_feat_map_fow.mean(1, keepdim=True)
        #global_feat_map_fow = global_feat_map.view(n, self.num_segments*c, h, w)#zhuyibunengfenka
        #global_feat_map_fow = global_feat_map.view(n, self.num_segments, c, h, w)#zhuyibunengfenka
        global_feat_map_fow = self.action_p1_squeeze1(global_feat_map_fow)
        global_feat_map_fow = global_feat_map_fow.view(n,self.num_segments, 1, h, w).transpose(2,1).contiguous()
        
        #global_feat_map_fow = global_feat_map.view(n, self.num_segments, 2048, 7, 7).transpose(2,1).contiguous()#zhuyibunengfenka
        global_feat_map_fow = self.action_p1_conv1(global_feat_map_fow)
        global_feat_map_fow = global_feat_map_fow.transpose(2,1).contiguous().view(n*self.num_segments, 1, h, w)
        global_feat_map_fow=self.sigmoid(global_feat_map_fow)
        #global_feat_map_fow=global_feat_map_fow*global_feat_map+global_feat_map
        global_feat_map_fow=global_feat_map_fow*global_feat_map.detach()
        
               
        
        #global_feat_map=global_feat_map.mean(1, keepdim=True)
        
        
        
        x3 = self.action_p3_squeeze(global_feat_map.detach())
        x3 = self.action_p3_bn1(x3)
        nt, c, h, w = x3.size()
        n_batch=nt//self.num_segments
        x3_plus0, _ = x3.view(n_batch, self.num_segments, c, h, w).split([self.num_segments-1, 1], dim=1)
        x3_plus1 = self.action_p3_conv1(x3)
        _ , x3_plus1 = x3_plus1.view(n_batch, self.num_segments, c, h, w).split([1, self.num_segments-1], dim=1)
        x_p3 = x3_plus1 - x3_plus0
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)
        x_p3=x_p3.view(nt, c, h, w)
        #x_p3 = self.avg_pool(x_p3.view(nt, c, h, w))
        #x_p3 = self.action_p3_expand(x_p3)
        x_p3 = self.sigmoid(x_p3)
        #x_p3 = global_feat_map * x_p3 +global_feat_map
        x_p3 = global_feat_map.detach() * x_p3
        
        #global_feat_map_total=x_p3+global_feat_map_fow+global_feat_map.detach()
        global_feat_map_total=x_p3+global_feat_map_fow
        #global_feat_map_total=self.sigmoid(x_p3+global_feat_map_fow)
        #global_feat_map_total=self.sigmoid(x_p3+global_feat_map_fow)*global_feat_map.detach()+global_feat_map.detach()
        global_feat_map_total=global_feat_map_total.mean(1, keepdim=True)
        #global_feat_map_total_x=global_feat_map_total.mean(2,keepdim=True)
        #global_feat_map_total_x=F.normalize(global_feat_map_total_x,p=1,dim=3)
        #global_feat_map_total_y=global_feat_map_total.mean(3,keepdim=True)
        #global_feat_map_total_y=F.normalize(global_feat_map_total_y,p=1,dim=2)

        #global_feat_map_total=F.softmax(global_feat_map_total,dim=2)
        #global_feat_map_total=global_feat_map_total.view(-1, 1, 7,7)
        #global_feat_map=global_feat_map.mean(1, keepdim=True)
        '''
        global_feat= global_feat.view(-1, self.global_feature_dim,1,1)
        global_feat_p2 = self.action_p2_squeeze_global(global_feat)
        nt, c, h, w = global_feat_p2.size()
        global_feat_p2 = global_feat_p2.view(-1, self.num_segments, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
        global_feat_p2 = self.action_p2_conv1_global(global_feat_p2)
        global_feat_p2 = self.relu(global_feat_p2)
        global_feat_p2= global_feat_p2.transpose(2,1).contiguous().view(-1, c, 1, 1)
        global_feat_p2 = self.action_p2_expand_global(global_feat_p2)
        global_feat_p2 = self.sigmoid(global_feat_p2)
        global_feat = global_feat * global_feat_p2 + global_feat
        global_feat = global_feat.view(-1, self.global_feature_dim)
        '''
        
        if(mode=='train'):
        #global_feat_map=global_feat_map.view(n*self.num_segments, 1, 7*7)
        #global_feat_map=F.softmax(global_feat_map,dim=2)
        #global_feat_map=global_feat_map.view(n*self.num_segments, 1, 7,7)
        
            #actions_1 = self.policy_stn(global_feat_map.detach())
            #actions_2 = torch.rand_like(actions_1)
            a=random.random()
            if (a>0.5):
             global_feat_map_1 = torch.rand_like(global_feat_map_total)
             #global_feat_map_1_x=global_feat_map_1.mean(2,keepdim=True)
             #global_feat_map_1_x=F.normalize(global_feat_map_1_x,p=1,dim=3)
             #global_feat_map_1_y=global_feat_map_1.mean(3,keepdim=True)
             #global_feat_map_1_y=F.normalize(global_feat_map_1_y,p=1,dim=2)
             #global_feat_map_1=global_feat_map_1.view(-1, 1, 7*7)#this is very important
             #global_feat_map_1=F.softmax(global_feat_map_1,dim=2)
             #global_feat_map_1=global_feat_map_1.view(-1, 1, 7,7)
             #patches = get_patches_frame(images, global_feat_map_1_x, global_feat_map_1_y, self.patch_size, self.input_size)
             patches = get_patches_frame(images, global_feat_map_1, self.patch_size, self.input_size)
            #res = []
            #for actions in [actions_1, actions_2]:
            else:
             #patches = get_patches_frame(images, global_feat_map_total_x, global_feat_map_total_y, self.patch_size, self.input_size)
             patches = get_patches_frame(images, global_feat_map_total, self.patch_size, self.input_size)
            local_feat = self.local_CNN.get_featvec(patches)
            #local_logits = self.local_CNN_fc(local_feat)
            #xiugaile
            '''
            local_feat= local_feat.view(-1, self.local_feature_dim,1,1)
            local_feat_p2 = self.action_p2_squeeze(local_feat)
            nt, c, h, w = local_feat_p2.size()
            local_feat_p2 = local_feat_p2.view(-1, self.num_segments, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
            local_feat_p2 = self.action_p2_conv1(local_feat_p2)
            local_feat_p2 = self.relu(local_feat_p2)
            local_feat_p2= local_feat_p2.transpose(2,1).contiguous().view(-1, c, 1, 1)
            local_feat_p2 = self.action_p2_expand(local_feat_p2)
            local_feat_p2 = self.sigmoid(local_feat_p2)
            local_feat = local_feat * local_feat_p2 + local_feat
            local_feat = local_feat.view(-1, self.local_feature_dim)
            '''
            cat_feat = torch.cat([global_feat, local_feat], dim=-1)
            
            #cat_feat = cat_feat.view(-1, self.num_segments, self.cat_feature_dim)
            #cat_logits, cat_pred = self.classifier(cat_feat)#xiugaishuchu
            #res.append((cat_feat,  global_logits, local_logits))
            return cat_feat,global_feat,local_feat
        else:
            #actions = self.policy_stn(global_feat_map)
            #patches = get_patches_frame(images, global_feat_map_total_x,global_feat_map_total_y,self.patch_size, self.input_size)
            patches = get_patches_frame(images, global_feat_map_total,self.patch_size, self.input_size)
            local_feat = self.local_CNN.get_featvec(patches)
            #local_logits = self.local_CNN_fc(local_feat)
            '''
            local_feat= local_feat.view(-1, self.local_feature_dim,1,1)
            local_feat_p2 = self.action_p2_squeeze(local_feat)
            nt, c, h, w = local_feat_p2.size()
            local_feat_p2 = local_feat_p2.view(-1, self.num_segments, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
            local_feat_p2 = self.action_p2_conv1(local_feat_p2)
            local_feat_p2 = self.relu(local_feat_p2)
            local_feat_p2= local_feat_p2.transpose(2,1).contiguous().view(-1, c, 1, 1)
            local_feat_p2 = self.action_p2_expand(local_feat_p2)
            local_feat_p2 = self.sigmoid(local_feat_p2)
            local_feat = local_feat * local_feat_p2 + local_feat
            local_feat = local_feat.view(-1, self.local_feature_dim)
            '''
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
        return [{'params': self.action_p1_conv1.parameters(), 'lr_mult': 1, 'decay_mult': 1,
                 'name': "action_p1_conv1"}] \
               +[{'params': self.action_p3_squeeze.parameters(), 'lr_mult': 1, 'decay_mult': 1,
                 'name': "action_p3_squeeze"}] \
               +[{'params': self.action_p3_conv1.parameters(), 'lr_mult': 1, 'decay_mult': 1,
                 'name': "action_p3_conv1"}] \
               +[{'params': self.action_p1_squeeze1.parameters(), 'lr_mult': 1, 'decay_mult': 1,
                 'name': "action_p1_squeeze1"}] \
               +[{'params': self.action_p2_squeeze.parameters(), 'lr_mult': 1, 'decay_mult': 1,
                 'name': "action_p2_squeeze"}] \
               +[{'params': self.action_p2_conv1.parameters(), 'lr_mult': 1, 'decay_mult': 1,
                 'name': "action_p2_conv1"}] \
               +[{'params': self.action_p2_expand.parameters(), 'lr_mult': 1, 'decay_mult': 1,
                 'name': "action_p2_expand"}] \
               +[{'params': self.action_p2_squeeze_global.parameters(), 'lr_mult': 1, 'decay_mult': 1,
                 'name': "action_p2_squeeze_global"}] \
               +[{'params': self.action_p2_conv1_global.parameters(), 'lr_mult': 1, 'decay_mult': 1,
                 'name': "action_p2_conv1_global"}] \
               +[{'params': self.action_p2_expand_global.parameters(), 'lr_mult': 1, 'decay_mult': 1,
                 'name': "action_p2_expand_global"}] \
               +[{'params': self.local_CNN.parameters(), 'lr_mult': 1, 'decay_mult': 1, 'name': "local_CNN"}] \
               +[{'params': self.global_CNN.parameters(), 'lr_mult': 1, 'decay_mult': 1,
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
