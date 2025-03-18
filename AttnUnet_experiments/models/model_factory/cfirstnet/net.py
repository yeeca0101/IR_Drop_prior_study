import torch.nn as nn
import torch.nn.functional as F

from timm import create_model

from .layers import AddCoords, Conv2dNormAct

class Head(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()
        self.in_channels = encoder_channels
        self.channels = decoder_channels
        
        self.layer_convs = nn.ModuleList()
        for in_channels in self.in_channels:
            layer_conv = nn.Conv2d(in_channels, self.channels, kernel_size=1)
            self.layer_convs.append(layer_conv)
            
        self.final_conv = nn.Sequential(
            Conv2dNormAct(decoder_channels, decoder_channels, kernel_size=7, padding=3, dw_conv=True),
            Conv2dNormAct(decoder_channels, decoder_channels, kernel_size=7, padding=3, dw_conv=True),
            Conv2dNormAct(decoder_channels, decoder_channels, kernel_size=7, padding=3, dw_conv=True),
            Conv2dNormAct(decoder_channels, decoder_channels, kernel_size=7, padding=3, dw_conv=True),
        )

    def forward(self, inputs):
        # inputs -> layers
        layers = [layer_conv(inputs[i]) for i, layer_conv in enumerate(self.layer_convs)]
        
        # layers -> fpn_layers
        for i in reversed(range(0, len(layers) - 1)):
            layers[i] = layers[i] + F.interpolate(layers[i + 1], size=layers[i].shape[2:], mode="bilinear")
        
        return layers[0] + self.final_conv(layers[0])

class Net(nn.Module):
    def __init__(
        self,
        model_backbone: str,
        model_pretrained: bool,
        in_channels: int,
        stochastic_depth: float,
        dropout: float,
        decoder_channels: int,
        out_channels=1,
    ):
        super().__init__()
        self.addcoords = AddCoords(rank=2, with_r=True)
        self.shallow = nn.Conv2d(in_channels + 3, 64, kernel_size=3, padding=1)
        
        self.encoder = create_model(
            model_name=model_backbone, 
            pretrained=model_pretrained,
            drop_path_rate=stochastic_depth,
            features_only=True,
            in_chans=64,
        )
        encoder_channels = [64] + [info["num_chs"] for info in self.encoder.feature_info]
        
        self.decoder = Head(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        )
        
        self.drop = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(decoder_channels, out_channels, kernel_size=3, padding=1)
        
        self.aux_classifier = nn.Conv2d(decoder_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.shallow(x)
        x = [x] + self.encoder(x)
        features = self.decoder(x)
        features = self.drop(features)
        out = self.classifier(features)
        aux_out = self.aux_classifier(features)
        
        return {'x_recon':out, 'loc':aux_out}
    
if __name__ == '__main__':
    import torch


    model = Net(
        model_backbone="convnextv2_tiny.fcmae",
        model_pretrained='True',
        in_channels=25,
        stochastic_depth=0.5,
        dropout=0.0,
        decoder_channels=128,
        out_channels=1,
    )
    model.to('cuda:0')
    print(model(torch.randn((1,25,256,256)).to('cuda:0'))[0].shape)
