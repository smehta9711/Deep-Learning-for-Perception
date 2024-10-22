
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoConvLayerWithSkipConn(nn.Module):  # deriving from torch.nn base class
    
    def __init__(self,in_channels,out_channels, stride=1):  # constructor initializes the class and takes 2 arguments
        
        super(TwoConvLayerWithSkipConn,self).__init__()   # calls the constructor of parent class to initilaze our custom class
        
        # self.doubleconv=nn.Sequential(
        #     nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=kernel_size//2),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),   # ensures operation takes place here itself without allocating memory for output
        #     nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding=kernel_size//2),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        #     )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        # return self.doubleconv(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# class Encoder(nn.module):  # we will assign a double convolution layer followed by max/average pooling
    
#     def __init__(self,in_channels,out_channels):
        
#         super(Encoder,self).__init__()
        
#         self.encoder = nn.Sequential(
#             TwoConvLayerWithSkipConn(in_channels,out_channels),
#             nn.MaxPool2d(kernel_size=2, stride=2)   # reduce size by max pooling
#         )
        
#     def forward(self,x):
#         return self.encoder(x)


# class Decoder(nn.module):  # we will assign a double convolution layer followed by max/average pooling
    
#     def __init__(self,in_channels,out_channels):
        
#         super(Decoder,self).__init__()
        
#         self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)  # upsample by same amount to preserve spatial diensions
#         self.doubleconv = TwoConvLayerWithSkipConn(in_channels, out_channels)
   
#     def forward(self,x1,x2):
#         x1= self.upconv(x1)    # upsampled feature map 
        
#         if x1.size() != x2.size():
#             x1 = F.interpolate(x1, size=x2.shape[2:])

#         x = torch.cat([x2, x1], dim=1)   # This feature map (x2) captures important high-resolution details from the input image and is essential for later stages in the network.
#         # skip connections parameter will be defined in the Network class
#         # x = self.double_conv(x)
#         x = self.doubleconv(x)
        
#         return x



class Network(nn.Module):
    
    def __init__(self,num_channels,num_classes):
        
        super(Network,self).__init__()
        
        #self.enco1 = TwoConvLayer(num_channels,64)  # output size will be 256x256 for same input dim
        self.enco1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Input is RGB (3 channels)     # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )                                           # 64x64
                    #Following the basic resnet18 architecture with a single max pooling layer, with skip connection within encoder network.
                    # Kernel size = 7, padding = 3, stride = 2
                    # Reduces image size to 128x128, followed by max-pooling layer makes it 64x64
                    # With stride = 2, skip connection within encoder.
        self.enco2 = TwoConvLayerWithSkipConn(64,64)               # 64x64
        self.enco3 = TwoConvLayerWithSkipConn(64,128,stride=2)      # 32x32
        self.enco4 = TwoConvLayerWithSkipConn(128,256,stride=2)
        self.enco5 = TwoConvLayerWithSkipConn(256,512,stride=2)

        self.deco4 = self.decoderWithSkipConnections(512, 256)
        self.deco3 = self.decoderWithSkipConnections(256, 128)
        self.deco2 = self.decoderWithSkipConnections(128, 64)
        self.deco1 = self.decoderWithSkipConnections(64, 64)

        self.finalconv = nn.Conv2d(64,num_classes,kernel_size=1)  # kernel_size=1 1x1 convolution to the number of classes


    def decoderWithSkipConnections(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        # Encoder part
        e1 = self.enco1(x)  # First block output (128x128)
        e2 = self.enco2(e1)  # Second block output (64x64)
        e3 = self.enco3(e2)  # Third block output (32x32)
        e4 = self.enco4(e3)  # Fourth block output (16x16)
        e5 = self.enco5(e4)  # Fifth block output (8x8)
        
        # Decoder part with skip connections
        d4 = self.deco4(e5)  # Upsample (8x8 -> 16x16)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)  # Ensure size matches e4
        d4 = d4 + e4  # Skip connection with e4

        d3 = self.deco3(d4)  # Upsample (16x16 -> 32x32)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)  # Ensure size matches e3
        d3 = d3 + e3  # Skip connection with e3

        d2 = self.deco2(d3)  # Upsample (32x32 -> 64x64)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)  # Ensure size matches e2
        d2 = d2 + e2  # Skip connection with e2

        d1 = self.deco1(d2)  # Upsample (64x64 -> 128x128)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)  # Ensure size matches e1
        d1 = d1 + e1  # Skip connection with e1

        # Final output layer (binary mask, 128x128 -> 256x256)
        out = self.finalconv(d1)
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)  # Ensure final output is 256x256
        return out

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2, logits=True, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce
#         if alpha is None:
#             self.alpha = torch.tensor([1.0, 1.0])  # Assuming binary classification
#         else:
#             self.alpha = torch.tensor(alpha)

#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#             probs = torch.sigmoid(inputs)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
#             probs = inputs

#         p_t = probs * targets + (1 - probs) * (1 - targets)
#         alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
#         focal_loss = alpha_t * (1 - p_t) ** self.gamma * BCE_loss

#         if self.reduce:
#             return focal_loss.mean()
#         else:
#             return focal_loss





