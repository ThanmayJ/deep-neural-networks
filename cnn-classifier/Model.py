import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, in_height:int, in_width:int, in_channels:int, activation:str, layer_filters_list:list, dense_features:int, out_classes:int, kernel_size:int, stride:int=1, padding:int=0, dropout:float=0.0, use_batchnorm:bool=True):
        super().__init__()

        self.activation = getattr(torch.nn, activation)()
        self.convnet = self._make_conv_layers([in_channels]+layer_filters_list, out_classes, kernel_size, stride, padding, use_batchnorm)
        convnet_out_features = self._get_convnet_out_features(in_channels,in_height,in_width)
        self.densenet = self._make_dense_layers(convnet_out_features, dense_features, out_classes, use_batchnorm)
        
        self.use_dropout = dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

        self.print_modules()
    
    
    def print_modules(self):
        modules = [list(self.convnet.modules())[0]] + [list(self.densenet.modules())[0]] + ([self.dropout] if bool(self.use_dropout) else [])
        print("-"*100)
        for i,module in enumerate(modules):
            print(f"({i}): {module}")
        print("-"*100)

    def _make_conv_layers(self, layer_channels_list:list, out_classes:int, kernel_size:int, stride:int, padding:int, use_batchnorm:bool):
        layers = []
        self.maxpool = nn.MaxPool2d(kernel_size,stride,padding)

        for i in range(len(layer_channels_list) - 1):
            in_channels = layer_channels_list[i]
            out_channels = layer_channels_list[i+1]

            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                *(nn.BatchNorm2d(out_channels),) * use_batchnorm,
                self.activation, 
                self.maxpool
                ]  
        
        return nn.Sequential(*layers)
    
    def _make_dense_layers(self, convnet_out_features:int, dense_features:int, out_classes:int, use_batchnorm:bool):
        self.convnet_out_features = convnet_out_features
        self.dense_features = dense_features

        layers = [nn.Linear(convnet_out_features, dense_features),
                  *(nn.BatchNorm1d(dense_features),) * use_batchnorm,
                  self.activation,
                  nn.Linear(dense_features, out_classes),
                *(nn.BatchNorm1d(out_classes),) * use_batchnorm]

        return nn.Sequential(*layers)
        
        
    
    def _get_convnet_out_features(self,in_channels:int, in_height:int, in_width:int):
        dummy_input = torch.zeros(1,in_channels,in_height,in_width)
        dummy_output = self.convnet(dummy_input)
        convnet_out_features = dummy_output.view(dummy_output.size(0), -1).size(1)
        
        return convnet_out_features

    def forward(self, x:torch.tensor):
        x = self.convnet(x)
        x = x.view(x.size(0), -1) # flatten the convnet output to pass into linear layer
        x = self.densenet(x)
        # Note that softmax is implicitly handled by the nn.CrossEntropyLoss, so we do not include it here.
        if self.use_dropout:
            x = self.dropout(x)
        return x




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print("Running on device", DEVICE)

    batch_size, in_channels, in_height, in_width = 1, 3, 32, 32
    out_classes = 10
    layer_filters_list=[64,64,64,128,128]
    model = ConvNet(in_height=in_height, in_width=in_width, in_channels=in_channels, activation='ReLU', layer_filters_list=layer_filters_list, dense_features=512, out_classes=out_classes, kernel_size=3, stride=1, padding=0, dropout=0.1, use_batchnorm=True)
    model = model.to(DEVICE)
    
    input = torch.randn(batch_size, in_channels, in_height, in_width).to(DEVICE)
    output = model(input)
    model.print_modules()
    print(input.shape, output.shape)