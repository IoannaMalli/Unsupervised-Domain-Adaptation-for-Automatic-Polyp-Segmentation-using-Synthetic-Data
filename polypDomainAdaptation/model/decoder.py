import torch
import torch.nn as nn
import torch.nn.functional as F

class SEPASPPModule(nn.Module):
    def __init__(self, dilations, in_channels, channels):
        super(SEPASPPModule, self).__init__()
        self.aspp_layers = nn.ModuleList()  # Use nn.ModuleList to hold depthwise separable convolutions
        for dilation in dilations:
            self.aspp_layers.append(
                nn.Sequential(
                    # Depthwise Convolution
                    nn.Conv2d(
                        in_channels, in_channels, kernel_size=3 if dilation > 1 else 1,
                        padding=dilation if dilation > 1 else 0, dilation=dilation,
                        groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    # Pointwise Convolution
                    nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        return [aspp_module(x) for aspp_module in self.aspp_layers]

class SEPASPPHead(nn.Module):
    def __init__(self, dilations=(1, 6, 12, 18), in_channels=256, channels=256, num_classes=21):
        super(SEPASPPHead, self).__init__()
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=in_channels, out_channels=channels,
                kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True))

        # Use SEPASPPModule here
        self.aspp_modules = SEPASPPModule(dilations, in_channels, channels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(len(dilations) * channels + channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        # Image Pooling
        img_pool = self.image_pool(inputs)
        img_pool = F.interpolate(img_pool, size=inputs.size()[2:], mode='bilinear', align_corners=False)

        # ASPP Modules
        aspp_outs = self.aspp_modules(inputs)
        aspp_outs.append(img_pool)

        # Concatenate and apply bottleneck
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)

        # Classification Head
        output = self.cls_seg(output)
        output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
        return output

class DAFormerHead(nn.Module):
    def __init__(self, in_channels, embed_dims, dilations, num_classes):
        super(DAFormerHead, self).__init__()

        # Embedding Layers for each input scale
        self.embed_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            )
            for in_ch, embed_dim in zip(in_channels, embed_dims)
        ])

        # Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(sum(embed_dims), embed_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims[-1]),
            nn.ReLU(inplace=True)
        )

        # ASPP Head
        self.decoder = SEPASPPHead(
            dilations=dilations,
            in_channels=embed_dims[-1],
            channels=256,
            num_classes=num_classes
        )

    def forward(self, inputs):
        # Apply embedding layers to multi-scale features

        # for i in range(len(inputs)):
        #     print(f"Input {i} shape: {inputs[i].shape}")
        embedded_features = [
            F.interpolate(embed_layer(x), size=inputs[0].shape[2:], mode='bilinear', align_corners=False)
            for x, embed_layer in zip(inputs, self.embed_layers)
        ]
        # for i in range(len(embedded_features)):
        #     print(f"Embedded feature {i} shape: {embedded_features[i].shape}")
        # Fuse features
        fused_features = torch.cat(embedded_features, dim=1)
        fused_features = self.fusion_layer(fused_features)
        #print(f"Fused features shape: {fused_features.shape}")
        # Decode with ASPP
        return self.decoder(fused_features)
