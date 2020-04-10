import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from lib.pspnet import PSPNet

class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True, truncated_blocks=2, model_type=None, dilation_blocks=1):
        super(DenseFeatureExtractionModule, self).__init__()

        numLayers = [3, 6, 4, 3]

        self.model_type = model_type

        if model_type == 'vgg16':
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2, stride=1),
                nn.Conv2d(256, 512, 3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            )
            self.num_channels = 512

        elif model_type == 'res50':
            model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(
                *list(model.children())[: -truncated_blocks-1]
            )

            if dilation_blocks:
                for i in range(1,dilation_blocks+1):
                    for j in range(1,numLayers[truncated_blocks+i-2]):
                        self.model[-i][j].conv2=nn.Conv2d(int(512/2**(truncated_blocks-2+i)),  
                                                          int(512/2**(truncated_blocks-2+i)),  
                                                          kernel_size=(3, 3), stride=(1, 1), 
                                                          padding=(2**(dilation_blocks+1-i), 2**(dilation_blocks+1-i)), 
                                                          dilation=(2**(dilation_blocks+1-i),2**(dilation_blocks+1-i)), 
                                                          bias=False)
                
                    if truncated_blocks+i < 5:
                        self.model[-i][0].conv2=nn.Conv2d(int(512/2**(truncated_blocks-2+i)), 
                                                        int(512/2**(truncated_blocks-2+i)), 
                                                        kernel_size=(3, 3), stride=(1, 1), 
                                                        padding=(1, 1), bias=False)
                        self.model[-i][0].downsample[0]=nn.Conv2d(int(1024/2**(truncated_blocks-2+i)), 
                                                                int(2048/2**(truncated_blocks-2+i)), 
                                                              kernel_size=(1, 1), stride=(1, 1), 
                                                              bias=False)
                    elif truncated_blocks+i == 5:
                        self.model[3]=nn.MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)
                    else:
                        print('You dilate too much')
            self.num_channels = int(4096/(2**truncated_blocks))

        elif model_type == 'res101':
            model = models.resnet101(pretrained=True)
            self.model = nn.Sequential(
                *list(model.children())[: -truncated_blocks-1]
            )
            self.num_channels = int(4096/(2**truncated_blocks))

            if dilation_blocks:
                for i in range(1,dilation_blocks+1):
                    for j in range(1,numLayers[truncated_blocks+i-2]):
                        self.model[-i][j].conv2=nn.Conv2d(int(512/2**(truncated_blocks-2+i)),  
                                                          int(512/2**(truncated_blocks-2+i)),  
                                                          kernel_size=(3, 3), stride=(1, 1), 
                                                          padding=(2**(dilation_blocks+1-i), 2**(dilation_blocks+1-i)), 
                                                          dilation=(2**(dilation_blocks+1-i),2**(dilation_blocks+1-i)), 
                                                          bias=False)
                
                    if truncated_blocks+i < 5:
                        self.model[-i][0].conv2=nn.Conv2d(int(512/2**(truncated_blocks-2+i)), 
                                                        int(512/2**(truncated_blocks-2+i)), 
                                                        kernel_size=(3, 3), stride=(1, 1), 
                                                        padding=(1, 1), bias=False)
                        self.model[-i][0].downsample[0]=nn.Conv2d(int(1024/2**(truncated_blocks-2+i)), 
                                                                int(2048/2**(truncated_blocks-2+i)), 
                                                              kernel_size=(1, 1), stride=(1, 1), 
                                                              bias=False)
                    
                    elif truncated_blocks+i == 5:
                        self.model[3]=nn.MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)
                    else:
                        print('You dilate too much')

            self.num_channels = int(4096/(2**truncated_blocks))


        self.use_relu = use_relu
        
        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, batch):
        if self.model_type == 'vgg16':
            output = self.model(batch)
            if self.use_relu:
                output = F.relu(output)
        elif self.model_type == 'res50':
            output = self.model(batch)
        elif self.model_type == 'res101':
            output = self.model(batch)            
        return output


class D2Net(nn.Module):
    def __init__(self, model_file=None, use_relu=True, use_cuda=True, truncated_blocks=2, model_type=None, edge_threshold=5, dilation_blocks=1, mode=1, pspnetTest=0):
        super(D2Net, self).__init__()

        if pspnetTest:
            self.dense_feature_extraction = PSPNet(
                n_classes=19, 
                pyramids=[6, 3, 2, 1], 
                input_size=[713, 713],
                use_bn=True, 
                output_features=False, 
                output_all=False, 
                model_type=model_type, 
                truncated_blocks=truncated_blocks, 
                dilation_blocks=dilation_blocks, 
                d2netTest=True
            )
        else:
            self.dense_feature_extraction = DenseFeatureExtractionModule(
                use_relu=use_relu, 
                use_cuda=use_cuda,
                truncated_blocks=truncated_blocks,
                model_type=model_type,
                dilation_blocks=dilation_blocks
            )

        self.detection = HardDetectionModule(edge_threshold=edge_threshold)

        self.localization = HandcraftedLocalizationModule()

        if model_file is not None:
            if pspnetTest:
                if use_cuda:
                    self.load_state_dict(torch.load(model_file)['state_dict'], strict=False)
                else:
                    self.load_state_dict(torch.load(model_file, map_location='cpu')['state_dict'], strict=False)
            else:
                if use_cuda:
                    self.load_state_dict(torch.load(model_file)['model'])
                else:
                    self.load_state_dict(torch.load(model_file, map_location='cpu')['model'])

    def forward(self, batch):
        _, _, h, w = batch.size()
        dense_features = self.dense_feature_extraction(batch)

        detections = self.detection(dense_features)

        displacements = self.localization(dense_features)

        return {
            'dense_features': dense_features,
            'detections': detections,
            'displacements': displacements
        }


class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)
        del local_max

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected


class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor(
            [[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]
        ).view(1, 1, 3, 3)
        self.dj_filter = torch.tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
        ).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)
        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        di = F.conv2d(
            batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1
        ).view(b, c, h, w)
        dj = F.conv2d(
            batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1
        ).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        return torch.stack([step_i, step_j], dim=1)
