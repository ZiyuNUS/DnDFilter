import torch.nn as nn

class NoMaD(nn.Module):

    def __init__(self, vision_encoder, 
                       noise_pred_net,):
        super(NoMaD, self).__init__()
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net

    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder" :
            output = self.vision_encoder(kwargs["obs_img"])
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        return output





