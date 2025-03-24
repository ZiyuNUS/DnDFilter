import torch.nn as nn

class DnD(nn.Module):

    def __init__(self, vision_encoder, noise_pred_net, state_extractor = None, fusion_enhancer = None):
        super(DnD, self).__init__()
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.state_extractor = state_extractor
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder" :
            output = self.vision_encoder(kwargs["obs_img"])
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "state_extractor":
            output = self.state_extractor(kwargs["obs_img"], kwargs["state_vector"])
        return output





