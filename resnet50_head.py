from torchvision import models
import torch.nn as nn

class Resnet50Head(nn.Module):
    def __init__(self):
        super(Resnet50Head, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.activation = {}
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
    
    def forward(self, x):
        
        
        self.resnet50.layer1.register_forward_hook(self.get_activation('conv2_3'))
        self.resnet50.layer2.register_forward_hook(self.get_activation('conv3_4'))
        self.resnet50.layer3.register_forward_hook(self.get_activation('conv4_6'))
        self.resnet50.layer4.register_forward_hook(self.get_activation('conv5_3'))
        
        x = self.resnet50(x)
        
        return [self.activation['conv2_3'],
                self.activation['conv3_4'],
                self.activation['conv4_6'],
                self.activation['conv5_3']]