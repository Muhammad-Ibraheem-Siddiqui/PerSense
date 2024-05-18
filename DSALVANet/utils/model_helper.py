from DSALVANet.models.DSALVANet import build_network
import torch
def build_model(weight_path,device):
    model = build_network(n_block=4,pool=[3,3],embed_dim=256,mid_dim=256,sample_times=1,dropout=0,activation='leaky_relu')
    model=model.to(device)
    checkpoint= torch.load(weight_path)
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model