import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ColorizationNet(nn.Module):
    def __init__(self, midlevel_input_size=128, global_input_size=512):
        super(ColorizationNet, self).__init__()
        # Fusion layer to combine midlevel and global features
        self.midlevel_input_size = midlevel_input_size
        self.global_input_size = global_input_size
        self.fusion = nn.Linear(midlevel_input_size + global_input_size, midlevel_input_size)
        self.bn1 = nn.BatchNorm1d(midlevel_input_size)

        # Convolutional layers and upsampling
        self.deconv1_new = nn.ConvTranspose2d(midlevel_input_size, 128, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(midlevel_input_size, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)

        print('Loaded colorization net.')

    def forward(self, midlevel_input): #, global_input):
        
        # Convolutional layers and upsampling
        x = F.relu(self.bn2(self.conv1(midlevel_input)))
        x = self.upsample(x)
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = F.sigmoid(self.conv4(x))
        x = self.upsample(self.conv5(x))
        return x


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        
        # Build ResNet and change first conv layer to accept single-channel input
        resnet_gray_model = models.resnet18(num_classes=365)
        resnet_gray_model.conv1.weight = nn.Parameter(resnet_gray_model.conv1.weight.sum(dim=1).unsqueeze(1).data)
        
        # Only needed if not resuming from a checkpoint: load pretrained ResNet-gray model
        if torch.cuda.is_available(): # and only if gpu is available
            resnet_gray_weights = torch.load('pretrained/resnet_gray_weights.pth.tar') #torch.load('pretrained/resnet_gray.tar')['state_dict']
            resnet_gray_model.load_state_dict(resnet_gray_weights)
            print('Pretrained ResNet-gray weights loaded')

        # Extract midlevel and global features from ResNet-gray
        self.midlevel_resnet = nn.Sequential(*list(resnet_gray_model.children())[0:6])
        self.global_resnet = nn.Sequential(*list(resnet_gray_model.children())[0:9])
        self.fusion_and_colorization_net = ColorizationNet()

    def forward(self, input_image):

        # Pass input through ResNet-gray to extract features
        midlevel_output = self.midlevel_resnet(input_image)
        # global_output = self.global_resnet(input_image)

        # Combine features in fusion layer and upsample
        output = self.fusion_and_colorization_net(midlevel_output) #, global_output)
        return output
