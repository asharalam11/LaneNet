import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SCNN(nn.Module):
    def __init__(
            self,
            input_size,
            ms_ks=9,
            pretrained=True
    ):
        """
        Argument
            ms_ks: kernel size in message passing conv
        """
        super(SCNN, self).__init__()
        self.pretrained = pretrained
        self.net_init(input_size, ms_ks)
        if not pretrained:
            self.weight_init()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None, exist_gt=None):
        x = self.backbone(img)
        x = self.layer1(x)
        x = self.message_passing_forward(x)
        x = self.layer2(x)

        seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        x = self.layer3(x)
        x = x.view(-1, self.fc_input_feature)
        exist_pred = self.fc(x)

        if seg_gt is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, exist_pred, loss_seg, loss_exist, loss

    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def net_init(self, input_size, ms_ks):
        input_w, input_h = input_size
        self.fc_input_feature = 5 * int(input_w/16) * int(input_h/16)
        
        ################################## SELECTING BACKBONE ARCHITECTURE ########################################
        self.backbone = self.select_backbone('resnet') # ResNet18 for now
        ###############################################################################################
        
        
        # ----------------- SCNN part -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()  # (nB, 128, 36, 100)
        )

        # ----------------- add message passing -----------------
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        # (nB, 128, 36, 100)

        # ----------------- SCNN part -----------------
        self.layer2 = nn.Sequential(
            #nn.Dropout2d(0.1),
            nn.Dropout2d(0.5),
            nn.Conv2d(128, 5, 1)  # get (nB, 5, 36, 100)
        )

        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),  # (nB, 5, 36, 100)
            nn.AvgPool2d(2, 2),  # (nB, 5, 18, 50)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def weight_init(self):
        print("Initializing weights")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
                #nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
                #m.weight.data[:] = 1.
                #m.bias.data.zero_()
                #nn.init.normal_(m.weight)
                #nn.init.zeros_(m.bias)
                
    # Selecting model backbone from various vision models trained on ImageNet          
    def select_backbone(self, model):
        print("wtf")
        if (model == 'resnet'):
            model = models.resnet18(pretrained=True)
            ## Extracting the model layers as elementst of a list
            mod = list(model.children())
            # Removing all layers after layer 33
            #for i in range(33):
            mod.pop()
            mod.pop()
            mod.pop()
            mod.pop()
            convolutional = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
            relu = nn.ReLU(inplace = True)
            model = torch.nn.Sequential(*mod, convolutional, relu)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
            model = model.to(device)
        elif (model == 'googlenet'):
            model = models.googlenet(pretrained=True)
            ## Extracting the model layers as elementst of a list
            mod = list(model.children())
            # Removing all layers after layer 33
            for i in range(12):
              mod.pop()
            conv = nn.Conv2d(480, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            bn = nn.BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
            relu = nn.ReLU(inplace=True)
            model = torch.nn.Sequential(*mod, conv, bn, relu)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
            model = model.to(device)
        elif (model == 'shufflenet'):
            ## Extracting the model layers as elementst of a list
            mod = list(model.children())
            # Removing all layers after layer 33
            for i in range(4):
              mod.pop()
            conv = nn.Conv2d(116, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            bn = nn.BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            relu = nn.ReLU(inplace=True)
            model = torch.nn.Sequential(*mod, conv, relu)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
            model = model.to(device)
        elif (model == 'mobilenet'):
            model = models.mobilenet_v2(pretrained=True).features
            ## Extracting the model layers as elementst of a list
            mod = list(model.children())
            # Removing all layers after layer 33
            for i in range(12):
              mod.pop()
            convolutional = nn.Conv2d(32, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
            relu = nn.ReLU6(inplace = True)
            model = torch.nn.Sequential(*mod, convolutional, relu)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
            model = model.to(device)
        elif (model == 'resnext'):
            model = models.resnext50_32x4d(pretrained=True)
            ## Extracting the model layers as elementst of a list
            mod = list(model.children())
            # Removing all layers after layer 33
            for i in range(4):
              mod.pop()
            conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            bn =  nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            relu = nn.ReLU(inplace = True)
            model = torch.nn.Sequential(*mod, conv, bn, Relu)
            #model = torch.nn.Sequential(*mod)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
            model = model.to(device)
        elif (model == 'wideresnet'):
            model = models.wide_resnet50_2(pretrained=True)
            mod = list(model.children())
            # Removing all layers after layer 33
            for i in range(4):
              mod.pop()
            conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            bn =  nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            relu = nn.ReLU(inplace = True)
            model = torch.nn.Sequential(*mod, conv, bn, relu)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
            model = model.to(device)
        elif (model == 'mnasnet'):
            model = models.mnasnet1_0(pretrained=True).layers
            mod = list(model.children())
            # Removing all layers after layer 33
            for i in range(7):
              mod.pop()
            conv = nn.Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
            bn = nn.BatchNorm2d(240, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
            relu = nn.ReLU(inplace=True)
            conv1 = nn.Conv2d(240, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
            relu1 = nn.ReLU(inplace=True)
            model = torch.nn.Sequential(*mod, conv, bn, relu, conv1, bn1, relu1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
            model = model.to(device)
        elif (model == 'vgg16_bn'):
            model = models.vgg16_bn(pretrained=self.pretrained).features
            # ----------------- process backbone -----------------
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
            
            for i in [34, 37, 40]:
                conv = model._modules[str(i)]
                dilated_conv = nn.Conv2d(
                    conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                    padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
                )
                dilated_conv.load_state_dict(conv.state_dict())
                model._modules[str(i)] = dilated_conv
            model._modules.pop('33')
            model._modules.pop('43')
            model = model.to(device)
        # Outside the if - else if
        return model

            
            

