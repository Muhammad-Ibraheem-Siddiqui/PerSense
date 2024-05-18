

import torch
import torch.nn.functional as F
from torch import nn
import torchvision
import copy

class DSALVANet(nn.Module):
    def __init__(
        self,
        n_block,
        pool,
        embed_dim,
        mid_dim,
        sample_times,
        dropout,
        activation,
    ):
        super().__init__()
        self.backbone = ResNet(out_stride=4,out_layers=[1,2,3])
        self.in_conv = nn.Conv2d(self.backbone.out_dim, embed_dim, kernel_size=1, stride=1)
        self.dsam_blocks = DSAMMultiBlock(
            n_block=n_block,
            pool=pool,
            out_stride=self.backbone.out_stride,
            embed_dim=embed_dim,
            mid_dim=mid_dim,
            dropout=dropout,
            activation=activation,
        )
        self.lvam = LVAM(sample_times= sample_times  ,in_dim=embed_dim)
        self.count_regressor = Regressor(in_dim=embed_dim, activation=activation)

    def forward(self, image,boxes):
        feat = self.in_conv(self.backbone(image)) 
        output = self.dsam_blocks(feat_orig=feat,boxes_orig=boxes) 
        output, _, _ = self.lvam(output) 
        output = self.count_regressor(output) 
        return output

class ResNet(nn.Module):
    def __init__(self, out_stride, out_layers):
        super().__init__()
        self.out_stride = out_stride
        self.out_layers = out_layers
        base_dim = 256
        self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        children = list(self.resnet.children())
        self.layer0 = nn.Sequential(*children[:4])  
        self.layer1 = children[4]
        self.layer2 = children[5]
        self.layer3 = children[6]
        self.layer4 = children[7]
        planes = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.out_dim = sum([planes[i - 1] for i in self.out_layers])

    def forward(self, x):
        x = self.layer0(x)  
        feat1 = self.layer1(x)  
        feat2 = self.layer2(feat1) 
        feat3 = self.layer3(feat2)  
        feat4 = self.layer4(feat3)  
        feats = [feat1, feat2, feat3, feat4]
        out_strides = [4, 8, 16, 32]
        feat_list = []
        for i in self.out_layers:
            scale_factor = out_strides[i - 1] / self.out_stride
            feat = feats[i - 1]
            feat = F.interpolate(feat, scale_factor=scale_factor, mode="bilinear")
            feat_list.append(feat)
        feat = torch.cat(feat_list, dim=1)
        return feat

class DSAMMultiBlock(nn.Module):
    def __init__(
        self,
        n_block,
        pool,
        out_stride,
        embed_dim,
        mid_dim,
        dropout,
        activation,
    ):
        super().__init__()
        self.out_stride = out_stride
        self.pool = pool
        dsam_block = DSAM(
            pool=pool,
            embed_dim=embed_dim,
            mid_dim=mid_dim,
            dropout=dropout,
            activation=activation,
        )
        self.dsam_blocks = get_clones(dsam_block, n_block)

    def crop_roi_feat(self,feat, boxes, out_stride, exemplar_pool_size):
        _, _, h, w = feat.shape
        boxes_scaled = boxes / out_stride
        boxes_scaled[:, :, :2] = torch.floor(boxes_scaled[:, :, :2]) 
        boxes_scaled[:, :, 2:] = torch.ceil(boxes_scaled[:, :, 2:])  
        boxes_scaled[:, :, :2] = torch.clamp_min(boxes_scaled[:, :, :2], 0)
        boxes_scaled[:, :, 2] = torch.clamp_max(boxes_scaled[:, :, 2], h)
        boxes_scaled[:, :, 3] = torch.clamp_max(boxes_scaled[:, :, 3], w)
        batch_feat_boxes = []
        feat_boxes = []
        for jdx_batch in range(0, boxes.shape[0]):
            for idx_box in range(0, boxes.shape[1]):
                y_tl, x_tl, y_br, x_br = boxes_scaled[jdx_batch][idx_box]
                y_tl, x_tl, y_br, x_br = int(y_tl), int(x_tl), int(y_br), int(x_br)
                feat_box = feat[jdx_batch][:, y_tl : (y_br + 1), x_tl : (x_br + 1)] 
                feat_box = F.adaptive_max_pool2d(feat_box, exemplar_pool_size, return_indices=False) 
                feat_boxes.append(feat_box)
            feat_boxes_tensor = torch.cat(feat_boxes, dim=0).contiguous().view(-1,feat.shape[1],feat_box.shape[-1],feat_box.shape[-1])               
            batch_feat_boxes.append(feat_boxes_tensor)
            feat_boxes=[]
        return batch_feat_boxes

    def forward(self, feat_orig, boxes_orig):
        output = feat_orig
        for block in self.dsam_blocks:
            feat_boxes_list = self.crop_roi_feat(output, boxes_orig, self.out_stride,self.pool)
            output = block(output, feat_boxes_list)
        return output

class DSAM(nn.Module):
    def __init__(
        self,
        pool,
        embed_dim,
        mid_dim,
        dropout,
        activation,
    ):
        super().__init__()
        self.similarity_cal = Similarity_cal(pool, embed_dim, dropout)
        self.conv1 = nn.Conv2d(embed_dim, mid_dim, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(mid_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)()

    def forward(self, querys_feat, supports_feat_list):
        M_sq = self.similarity_cal(querys_feat, supports_feat_list) 
        querys_feat = querys_feat + self.dropout1(M_sq)
        querys_feat = querys_feat.permute(0, 2, 3, 1).contiguous()
        querys_feat = self.norm1(querys_feat).permute(0, 3, 1, 2).contiguous()
        M_sq = self.conv2(self.dropout(self.activation(self.conv1(querys_feat))))
        M_sq = querys_feat + self.dropout2(M_sq)
        M_sq = M_sq.permute(0, 2, 3, 1).contiguous()
        M_sq = self.norm2(M_sq).permute(0, 3, 1, 2).contiguous()
        return M_sq

class Similarity_cal(nn.Module):  
    def __init__(self, pool, embed_dim, dropout, group=2):
        super().__init__()
        assert pool[0] % 2 == 1 and pool[1] % 2 == 1 
        self.pool = pool
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.in_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.out_conv = nn.Conv2d(group*3*4, embed_dim, kernel_size=1, stride=1)
        self.group = group
        self.mdm_d1 = MDM(in_channels=embed_dim , 
                                        out_channels=group * 3,
                                        kernel_size=pool[-1],
                                        offset_groups=8,
                                        dilation=1, 
                                        groups = group,                           
                                        with_bias=False,
                                        with_mask=True)
        self.mdm_d2 = MDM(in_channels=embed_dim , 
                                        out_channels=group * 3,
                                        kernel_size=pool[-1],
                                        offset_groups=8,
                                        dilation=2, 
                                        groups = group,                           
                                        with_bias=False,
                                        with_mask=True)
        self.mdm_d4 = MDM(in_channels=embed_dim , 
                                        out_channels=group * 3,
                                        kernel_size=pool[-1],
                                        offset_groups=8,
                                        dilation=4, 
                                        groups = group,                           
                                        with_bias=False,
                                        with_mask=True)

    def forward(self, querys_feat, supports_feat_list):
        h_p, w_p = self.pool
        batch_size = querys_feat.shape[0]
        querys_feat = self.in_conv(querys_feat)
        querys_feat = querys_feat.permute(0, 2, 3, 1).contiguous() 
        querys_feat = self.norm(querys_feat).permute(0, 3, 1, 2).contiguous() 
        supports_feat = torch.cat(supports_feat_list,dim=0).view(batch_size,-1,self.embed_dim, h_p,w_p)   
        b_similarity_list = []
        for query_feat, support_feat in zip(querys_feat, supports_feat):
            query_feat = query_feat.unsqueeze(0) 
            support_feat = self.in_conv(support_feat) 
            support_feat = support_feat.permute(0, 2, 3, 1).contiguous()
            support_feat = self.norm(support_feat).permute(0, 3, 1, 2).contiguous()
            support_feat = support_feat.contiguous().view(-1, self.embed_dim, h_p, w_p) 
            similarity_list=[]
            similarity_list.append(F.conv2d(query_feat,
                                      F.adaptive_max_pool2d(support_feat, 1, return_indices=False).view(self.group*support_feat.size(0),-1,1,1),
                                      groups=self.group)) 
            similarity_list.append(self.mdm_d1(query_feat, support_feat))
            similarity_list.append(self.mdm_d2(query_feat, support_feat)) 
            similarity_list.append(self.mdm_d4(query_feat,support_feat))   
            similarity=torch.concat(similarity_list,dim=1)  
            b_similarity_list.append(similarity)
        b_similarity = torch.cat(b_similarity_list,dim=0) 
        b_similarity = self.out_conv(b_similarity)
        return b_similarity

class MDM(nn.Module):
    def __init__(self, 
                in_channels: int, 
                out_channels: int,
                offset_groups: int = 1, 
                kernel_size: int = 3, 
                dilation: int = 1 , 
                groups: int = 1,
                with_bias: bool =  False , 
                with_mask: bool = False):
        super(MDM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.offset_groups = offset_groups
        self.dilation = dilation
        self.with_bias = with_bias
        self.with_mask = with_mask
        self.groups = groups
        self.reset_parameters()
        assert kernel_size%2 != 0 
        assert dilation==1 or dilation%2==0 
        
    def reset_parameters(self):
        self.conv_offset = nn.Conv2d(self.in_channels,2*self.offset_groups*self.kernel_size**2,
                                        kernel_size=self.kernel_size, 
                                        stride=1, 
                                        padding=(self.kernel_size - 1)//2) 
        init_offset = torch.Tensor(torch.zeros([2*self.offset_groups*self.kernel_size**2,self.in_channels,self.kernel_size,self.kernel_size]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset) 

        if self.with_mask != False:
            self.conv_mask = nn.Conv2d(self.in_channels, self.offset_groups*self.kernel_size**2, 
                                        kernel_size=self.kernel_size, 
                                        stride=1,
                                        padding=(self.kernel_size - 1)//2) 
            init_mask = torch.Tensor(torch.zeros([self.offset_groups*self.kernel_size**2,self.in_channels,self.kernel_size,self.kernel_size])) + 0.5
            self.conv_mask.weight = torch.nn.Parameter(init_mask)   
        if self.with_bias != False:
            self.bias = torch.Tensor(torch.zeros(self.out_channels))
        else:self.bias = None

    def forward(self,x, conv_kernel):
        n_exemplars = conv_kernel.shape[0]
        conv_kernel = conv_kernel.view(n_exemplars*self.groups,
                                    self.in_channels//self.groups,
                                    self.kernel_size,self.kernel_size)
        offset = self.conv_offset(x)
        if self.with_mask != False:
            mask = torch.softmax(self.conv_mask(x),dim=1) 
        else:
            mask = None
        padding = self.compute_padding()
        out = torchvision.ops.deform_conv2d(input = x, 
                                            offset=offset, 
                                            weight=conv_kernel, 
                                            mask = mask, 
                                            bias=self.bias,
                                            padding= padding,
                                            dilation=(self.dilation, self.dilation)) 
        return out

   
    def compute_padding(self):
        k = self.kernel_size
        d = self.dilation  
        padding = ((k-1)*d)//2
        return (padding,padding)

class LVAM(nn.Module):
    def __init__(self,  sample_times = 1, in_dim = 256, mid1_dim = 512 ,mid2_dim = 512, out_dim = 256):
        super(LVAM, self).__init__()
        self.conv1 = nn.Conv2d(in_dim,mid1_dim,kernel_size=3,stride=1, padding=1)
        self.ln1 = nn.LayerNorm(mid1_dim)
        self.conv2 = nn.Conv2d(mid1_dim,mid2_dim,kernel_size=3,stride=1,padding=1)
        self.ln2 = nn.LayerNorm(mid2_dim)
        self.mu = nn.Conv2d(mid2_dim,out_dim,kernel_size=1,stride=1)
        self.ln_mu = nn.LayerNorm(out_dim)
        self.logvar = nn.Conv2d(mid2_dim,out_dim,kernel_size=1,stride=1)
        self.ln_logvar = nn.LayerNorm(out_dim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.sample_times = sample_times
    
    def encode(self,x):
        x = self.conv1(x)
        x = self.ln1(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x = self.relu(x)
        x = self.conv2(x)
        x = self.ln2(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x = self.relu(x)
        mu = self.mu(x)
        mu = self.relu(self.ln_mu(mu.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous())
        logvar = self.logvar(x)
        logvar = self.relu(self.ln_logvar(logvar.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous())
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar.mul(0.5))
            epsilons = []
            for i in range(self.sample_times): 
                epsilon = torch.randn_like(std).cuda()                
                epsilons.append(epsilon.mul(std).add_(mu))
            epsilons = torch.concat(epsilons,dim=0)
            return epsilons
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Regressor(nn.Module):
    def __init__(self, in_dim, activation):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 5, padding=2),
            get_activation(activation)(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_dim // 2, in_dim // 4, 3, padding=1),
            get_activation(activation)(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_dim // 4, in_dim // 8, 1),
            get_activation(activation)(),
            nn.Conv2d(in_dim // 8, 1, 1),
            get_activation("relu")(),
        )

    def forward(self, x):
        return self.regressor(x)  

def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU
    if activation == "leaky_relu":
        return nn.LeakyReLU
    raise NotImplementedError

def build_network(**kwargs):
    return DSALVANet(**kwargs)



