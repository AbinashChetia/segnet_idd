import torch.nn as nn
import torchvision.models as models

class ConvReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, convs, **kwargs) -> None:
        super(EncoderBlock, self).__init__()
        layers = []
        for i in range(convs):
            layers.append(ConvReLU(in_c if i == 0 else out_c, out_c, **kwargs))
        self.layers = nn.Sequential(*layers)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.layers(x)
        x, ind = self.pool(x)
        return x, ind

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, convs, classification=False, **kwargs) -> None:
        super(DecoderBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        layers = []
        for i in range(convs):
            # Last layer in final block → 1x1 conv if classification True
            if i == convs - 1 and classification:
                layers.append(nn.Conv2d(in_c, out_c, kernel_size=1, padding=0))
            elif i == convs - 1:
                layers.append(ConvReLU(in_c, out_c, **kwargs))
            else:
                layers.append(ConvReLU(in_c, in_c, **kwargs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, ind):
        x = self.unpool(x, ind)
        x = self.layers(x)
        return x

class SegNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21, base_f=64) -> None:
        super(SegNet, self).__init__()
        # Encoder: VGG16‐BN topology [2,2,3,3,3]
        self.enc1 = EncoderBlock(in_channels,  base_f, convs=2)
        self.enc2 = EncoderBlock(base_f,      base_f*2, convs=2)
        self.enc3 = EncoderBlock(base_f*2,    base_f*4, convs=3)
        self.enc4 = EncoderBlock(base_f*4,    base_f*8, convs=3)
        self.enc5 = EncoderBlock(base_f*8,    base_f*8, convs=3)

        # Decoder: symmetric to encoder blocks
        self.dec5 = DecoderBlock(base_f*8,    base_f*8, convs=3)
        self.dec4 = DecoderBlock(base_f*8,    base_f*4, convs=3)
        self.dec3 = DecoderBlock(base_f*4,    base_f*2, convs=3)
        self.dec2 = DecoderBlock(base_f*2,    base_f,   convs=2)
        self.dec1 = DecoderBlock(base_f,      num_classes, convs=1, classification=True)

    def forward(self, x):
        e1, i1 = self.enc1(x)
        e2, i2 = self.enc2(e1)
        e3, i3 = self.enc3(e2)
        e4, i4 = self.enc4(e3)
        e5, i5 = self.enc5(e4)

        d5 = self.dec5(e5, i5)
        d4 = self.dec4(d5, i4)
        d3 = self.dec3(d4, i3)
        d2 = self.dec2(d3, i2)
        out = self.dec1(d2, i1)
        return out

def load_vgg16_bn_weights(segnet_model: SegNet):

    vgg16_bn = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
    pretrained_layers = list(vgg16_bn.features)

    segnet_convs = []
    for module in segnet_model.modules():
        if isinstance(module, nn.Conv2d):
            segnet_convs.append(module)

    vgg_conv_idx = 0
    for m in pretrained_layers:
        if isinstance(m, nn.Conv2d):
            segnet_layer = segnet_convs[vgg_conv_idx]
            segnet_layer.weight.data.copy_(m.weight.data)
            segnet_layer.bias.data  .copy_(m.bias.data)
            vgg_conv_idx += 1

    print(f"Loaded {vgg_conv_idx} conv layers from VGG16‑BN into SegNet encoder.")