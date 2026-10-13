# ultralytics/nn/modules/multimodal_yolo11.py

import torch.nn as nn
from backbone_yolo11 import YOLO11Backbone, YOLO11nBackbone
# from fusion import GatedFusion2D
from fusion import GatedFusion2D, CrossAttnFusion2D

from head_yolo11_fused import YOLO11HeadFused
from head_yolo11_hybrid import YOLO11HybridHeadFused
import torch
from conv import Conv
from block import C2PSA

# multimodal_yolo11_fixed.py
import torch
import torch.nn as nn
from backbone_yolo11 import YOLO11Backbone

class SparseLiDARBackbone(nn.Module):
    def __init__(self, ch_in=3):
        super().__init__()
        self.stem = nn.Sequential(
            # nn.Conv2d(ch_in, 64, 1, 1),
            nn.Conv2d(ch_in, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.backbone = YOLO11Backbone(ch_in=64)

    def forward(self, x):
        x = x / 255.0
        x = self.stem(x)
        return self.backbone(x)  # returns (p3, p4, p5)


# 69.7%
# ## simple method simple fusion kv camera and q lidar
# class CrossAttnFusion2D(nn.Module):
#     def __init__(self, in_channels_a, in_channels_b, out_channels, num_heads=4, dropout=0.1):
#         super().__init__()
#         # Project b to match a's channel dimension for attention
#         self.proj_b = nn.Conv2d(in_channels_b, in_channels_a, 1) if in_channels_a != in_channels_b else nn.Identity()

#         self.dropout_pre_attn = nn.Dropout2d(p=dropout)  # Spatial dropout before attention
#         self.dropout_post_attn = nn.Dropout(p=dropout)   # Standard dropout after attention

#         self.attention = nn.MultiheadAttention(embed_dim=in_channels_a, num_heads=num_heads, dropout=dropout)
#         self.conv_a = nn.Conv2d(in_channels_a, out_channels, 1)
#         self.conv_b = nn.Conv2d(in_channels_a, out_channels, 1)  # Changed from in_channels_b
#         self.norm = nn.InstanceNorm2d(out_channels)
#         self.relu = nn.SiLU()  # SiLU is better than ReLU for YOLO
    

#     # lidar is mein here 
#     def forward(self, a, b):
#         # Project b to match a's channels
#         b_proj = self.proj_b(b)

#         # ADD: Apply dropout to camera (a) and lidar (b_proj) BEFORE attention
#         a_dropped = self.dropout_pre_attn(a)
#         b_proj_dropped = self.dropout_pre_attn(b_proj)

#         n, c, h, w = a.shape
        
#         # Use dropped versions for attention
#         a_flat = a_dropped.view(n, c, h * w).permute(2, 0, 1)  # Changed: a → a_dropped
#         b_flat = b_proj_dropped.view(n, c, h * w).permute(2, 0, 1)  # Changed: b_proj → b_proj_dropped

#         # Cross-attention (has internal dropout too)
#         # attn_output, _ = self.attention(query=a_flat, key=b_flat, value=b_flat)
#         attn_output, _ = self.attention(
#                 query=b_flat,   # LiDAR
#                 key=a_flat,     # camera
#                 value=a_flat    # camera
#             )

#         # ADD: Apply dropout AFTER attention
#         attn_output = self.dropout_post_attn(attn_output)

#         attn_output = attn_output.permute(1, 2, 0).view(n, c, h, w)

#         # Combine and normalize
#         fused = self.conv_a(a) + self.conv_b(attn_output)
#         return self.norm(fused)

## complex lidar to camera and camera to lidar bidriectinal fusion 
# class CrossAttnFusion2D(nn.Module):
#     def __init__(self, in_channels_cam, in_channels_lidar, out_channels, num_heads=4, dropout=0.1):
#         super().__init__()
        
#         # 1. Project LiDAR to match Camera channels (if needed)
#         self.proj_lidar = nn.Conv2d(in_channels_lidar, in_channels_cam, 1) if in_channels_lidar != in_channels_cam else nn.Identity()

#         # 2. Define TWO SEPARATE attention modules
#         # Reason: The relationship "Cam looking at Lidar" is different from "Lidar looking at Cam"
#         self.attn_cam_focus = nn.MultiheadAttention(embed_dim=in_channels_cam, num_heads=num_heads, dropout=dropout)
#         self.attn_lidar_focus = nn.MultiheadAttention(embed_dim=in_channels_cam, num_heads=num_heads, dropout=dropout)

#         # Dropouts
#         self.dropout_pre = nn.Dropout2d(p=dropout)
        
#         # 3. Final Fusion Layer
#         # We will concatenate (Cam_Enhanced) + (Lidar_Enhanced), so input is 2 * channels
#         self.out_conv = nn.Conv2d(in_channels_cam * 2, out_channels, 1)
#         self.norm = nn.InstanceNorm2d(out_channels)
#         self.act = nn.SiLU()

#     def forward(self, cam, lidar):
#         # 1. Align dimensions
#         lidar_proj = self.proj_lidar(lidar)
        
#         # 2. Apply Spatial Dropout before flattening
#         cam_d = self.dropout_pre(cam)
#         lidar_d = self.dropout_pre(lidar_proj)

#         n, c, h, w = cam.shape

#         # 3. Flatten for Attention (Sequence Length, Batch, Channels)
#         # Sequence Length = H * W
#         src_cam = cam_d.view(n, c, h * w).permute(2, 0, 1)
#         src_lidar = lidar_d.view(n, c, h * w).permute(2, 0, 1)

#         # --- DIRECTION 1: Enhance Camera using LiDAR ---
#         # Q=Cam, K=LiDAR, V=LiDAR
#         attn_out_1, _ = self.attn_cam_focus(query=src_cam, key=src_lidar, value=src_lidar)
#         feat_1 = attn_out_1.permute(1, 2, 0).view(n, c, h, w)
        
#         # [CRITICAL STEP] RESIDUAL CONNECTION
#         # We add the result back to the original Camera input
#         feat_1 = cam + feat_1 

#         # --- DIRECTION 2: Enhance LiDAR using Camera ---
#         # Q=LiDAR, K=Cam, V=Cam
#         attn_out_2, _ = self.attn_lidar_focus(query=src_lidar, key=src_cam, value=src_cam)
#         feat_2 = attn_out_2.permute(1, 2, 0).view(n, c, h, w)
        
#         # [CRITICAL STEP] RESIDUAL CONNECTION
#         # We add the result back to the projected LiDAR input
#         feat_2 = lidar_proj + feat_2

#         # --- FUSE ---
#         # Concatenate the enhanced features
#         cat_feats = torch.cat([feat_1, feat_2], dim=1)
        
#         # Project to output
#         out = self.out_conv(cat_feats)
#         out = self.norm(out)
#         out = self.act(out)

#         return out

import torch
import torch.nn as nn

class CrossAttnFusion2D(nn.Module):
    def __init__(self, in_channels_cam, in_channels_lidar, out_channels, num_heads=4, dropout=0.1):
        super().__init__()
        
        # 1. Project LiDAR to match Camera channels
        self.proj_lidar = nn.Conv2d(in_channels_lidar, in_channels_cam, 1) if in_channels_lidar != in_channels_cam else nn.Identity()

        # 2. TWO SEPARATE attention modules (Crucial for convergence)
        self.attn_cam_focus = nn.MultiheadAttention(embed_dim=in_channels_cam, num_heads=num_heads, dropout=dropout)
        self.attn_lidar_focus = nn.MultiheadAttention(embed_dim=in_channels_cam, num_heads=num_heads, dropout=dropout)

        # Layer Norm before attention: aligns feature distributions across modalities
        self.norm_cam   = nn.LayerNorm(in_channels_cam)
        self.norm_lidar = nn.LayerNorm(in_channels_cam)  # applied after lidar projection

        # Dropouts: pre-attention spatial dropout capped at 0.1 (0.3 was too aggressive)
        self.dropout_pre = nn.Dropout2d(p=min(dropout, 0.1))
        self.dropout_pre_cat = nn.Dropout(p=dropout)
        
        # 3. Final Fusion Layer
        # Input is 2 * channels because we concatenate
        self.out_conv = nn.Conv2d(in_channels_cam * 2, out_channels, 1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, cam, lidar):
        # 1. Align dimensions
        lidar_proj = self.proj_lidar(lidar)
        
        # 2. Apply Spatial Dropout before flattening
        cam_d = self.dropout_pre(cam)
        lidar_d = self.dropout_pre(lidar_proj)

        n, c, h, w = cam.shape

        # 3. Flatten for Attention (Sequence Length, Batch, Channels)
        src_cam   = cam_d.view(n, c, h * w).permute(2, 0, 1)    # [L, B, C]
        src_lidar = lidar_d.view(n, c, h * w).permute(2, 0, 1)  # [L, B, C]

        # 4. Layer Norm before attention
        src_cam   = self.norm_cam(src_cam)
        src_lidar = self.norm_lidar(src_lidar)

        # --- BRANCH 1: Camera Context ---
        # Q=Cam, K=LiDAR, V=LiDAR
        attn_out_1, _ = self.attn_cam_focus(query=src_cam, key=src_lidar, value=src_lidar)
        feat_1 = attn_out_1.permute(1, 2, 0).view(n, c, h, w)
        
        # Residual Connection (Crucial for fixing NaN)
        cam_enhanced = cam + feat_1 

        # --- BRANCH 2: LiDAR Context ---
        # Q=LiDAR, K=Cam, V=Cam
        attn_out_2, _ = self.attn_lidar_focus(query=src_lidar, key=src_cam, value=src_cam)
        feat_2 = attn_out_2.permute(1, 2, 0).view(n, c, h, w)
        
        # Residual Connection (Crucial for fixing NaN)
        lidar_enhanced = lidar_proj + feat_2

        # ---------------------------------------------------------
        # NEW: Apply dropout before concatenation as requested
        # ---------------------------------------------------------
        cam_enhanced = self.dropout_pre_cat(cam_enhanced)
        lidar_enhanced = self.dropout_pre_cat(lidar_enhanced)

        # --- FUSE ---
        # Concatenate
        cat_feats = torch.cat([cam_enhanced, lidar_enhanced], dim=1)
        
        # Final Projection
        out = self.out_conv(cat_feats)
        out = self.norm(out)
        out = self.act(out)

        return out


class LearnableGatedBilinearFusion(nn.Module):
    def __init__(self, in_channels_a, in_channels_b, out_channels, reduction_ratio=4):
        super().__init__()
        
        # Ensure inputs have the same channel count for simplified interaction
        if in_channels_a != in_channels_b:
            # We enforce channel alignment here via projection, assuming they will be the same
            # for the interaction step.
            self.proj_a = nn.Conv2d(in_channels_a, out_channels, 1)
            self.proj_b = nn.Conv2d(in_channels_b, out_channels, 1)
        else:
            self.proj_a = nn.Identity()
            self.proj_b = nn.Identity()
            
        # The common channel count for the fusion steps
        fusion_channels = out_channels if in_channels_a != in_channels_b else in_channels_a
        
        # 1. Bilinear Interaction Layer (A * B)
        # This 1x1 Conv learns the multiplicative interaction matrix between A and B channels
        # Concatenate A and B for the input to the interaction layer
        self.bilinear_interaction = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, 1),
            nn.SiLU()
        )
        
        # 2. Gate Mechanism (Similar to Squeeze-and-Excitation or Gated Fusion)
        # Learn the importance of the interaction and the two inputs
        self.gate = nn.Sequential(
            nn.Conv2d(fusion_channels * 3, fusion_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // reduction_ratio, fusion_channels * 3, 1),
            nn.Sigmoid()
        )
        
        # Final projection to desired output channels if needed (only if using Identity above)
        self.final_proj = nn.Conv2d(fusion_channels, out_channels, 1) if fusion_channels != out_channels else nn.Identity()
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, a, b):
        # 1. Channel Alignment
        a = self.proj_a(a)
        b = self.proj_b(b)
        
        # 2. Compute Bilinear Interaction Term (A * B)
        # The 1x1 Conv on concatenated features implicitly learns the correlation/outer product
        # between feature vectors at each spatial location.
        x_concat = torch.cat([a, b], dim=1)
        x_bilinear = self.bilinear_interaction(x_concat) # (N, C, H, W)

        # 3. Create Fusion Input Tensor
        # This tensor contains all three components: A, B, and the Interaction term
        x_fusion = torch.cat([a, b, x_bilinear], dim=1) # (N, 3*C, H, W)
        
        # 4. Generate Gates
        # The gate learns three weights (g_a, g_b, g_inter) for the three components
        g = self.gate(x_fusion) # (N, 3*C, H, W), where C = fusion_channels
        
        # Split the gate tensor into three parts
        g_a, g_b, g_inter = torch.split(g, a.shape[1], dim=1)
        
        # 5. Apply Gates and Combine
        # This is the core LGBF operation: weighted sum of the original inputs and their interaction.
        fused = (g_a * a) + (g_b * b) + (g_inter * x_bilinear)
        
        # 6. Final Normalization and Projection
        return self.norm(self.final_proj(fused))




class GatedFusion(nn.Module):
    def __init__(self, ch, dropout_prob=0.0):
        super().__init__()
        self.gate = nn.Conv2d(ch * 2, 2, 1)   # modality logits
        self.norm = nn.GroupNorm(32, ch)      # or BatchNorm2d(ch)
        self.dropout_prob = dropout_prob

    def forward(self, a, b):
        if self.training and self.dropout_prob > 0:
            if torch.rand((), device=a.device) < self.dropout_prob: a = 0 * a
            if torch.rand((), device=b.device) < self.dropout_prob: b = 0 * b

        x = torch.cat([a, b], 1)
        g = torch.softmax(self.gate(x), dim=1)   # weights sum to 1
        ga, gb = g[:, 0:1], g[:, 1:2]
        return self.norm(ga * a + gb * b)


class AdaptiveWeightedFusion(nn.Module):
    """
    An adaptive fusion module that learns a normalized weight for each input
    (A and B) based on the overall feature content of both.
    """
    def __init__(self, in_channels_a, in_channels_b, out_channels):
        super().__init__()

        # Project inputs to a common channel space (C_out)
        self.proj_a = nn.Conv2d(in_channels_a, out_channels, 1)
        self.proj_b = nn.Conv2d(in_channels_b, out_channels, 1)

        # The input to the weight prediction network is the concatenation of the projected features
        weight_net_in_channels = out_channels * 2

        # Weight Prediction Network (WPN)
        # This network predicts a single weight (or two, normalized) for the entire feature map.
        self.weight_net = nn.Sequential(
            # Global Average Pooling to reduce spatial dimensions (H, W -> 1, 1)
            nn.AdaptiveAvgPool2d(1), 
            # 1x1 Conv to reduce complexity (similar to a fully connected layer)
            nn.Conv2d(weight_net_in_channels, weight_net_in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(weight_net_in_channels // 4),
            nn.ReLU(inplace=True),
            # Final 1x1 Conv to output 2 weights (one for A, one for B)
            nn.Conv2d(weight_net_in_channels // 4, 2, 1, bias=False),
            # Softmax to ensure weights sum to 1 and are non-negative
            nn.Softmax(dim=1) 
        )

    def forward(self, a, b):
        # 1. Project inputs to common dimension
        a_proj = self.proj_a(a)
        b_proj = self.proj_b(b)

        # 2. Concatenate and pass through the Weight Prediction Network (WPN)
        # Input to WPN: (N, 2*C_out, H, W)
        x_concat = torch.cat([a_proj, b_proj], dim=1) 
        
        # Output of WPN: (N, 2, 1, 1) -> containing [w_a, w_b]
        weights = self.weight_net(x_concat) 

        # 3. Extract and Reshape Weights
        # w_a, w_b are (N, 1, 1, 1) after splitting and squeezing
        w_a = weights[:, 0:1, :, :]
        w_b = weights[:, 1:2, :, :]

        # 4. Adaptive Fusion
        # Multiply each feature map by its learned weight
        fused = (w_a * a_proj) + (w_b * b_proj)

        return fused


class MultiModalYOLO11(nn.Module):
    def __init__(self, nc=10, cam_in_ch=3, lidar_in_ch=3, fusion_type="cross_attn"):
        super().__init__()
        # Hybrid: YOLOv11m for camera, YOLOv11n for LiDAR
        self.backbone_cam = YOLO11Backbone(ch_in=cam_in_ch)       # YOLOv11m: 768, 768, 1536
        self.backbone_lidar = YOLO11nBackbone(ch_in=lidar_in_ch)  # YOLOv11n: 512, 512, 1024

        self.fusion_type = fusion_type

        # Asymmetric fusion: Camera (768/768/1536) + LiDAR (512/512/1024) -> Output (768/768/1536)
        # mAP50 = 82.3 27.1
        if fusion_type == "gated":
            # GatedFusion expects same input channels, need to project LiDAR first
            self.lidar_proj_p3 = nn.Conv2d(512, 768, 1)    # Project LiDAR P3: 512->768
            self.lidar_proj_p4 = nn.Conv2d(512, 768, 1)    # Project LiDAR P4: 512->768
            self.lidar_proj_p5 = nn.Conv2d(1024, 1536, 1)  # Project LiDAR P5: 1024->1536

            self.fuse_p3 = GatedFusion(768)
            self.fuse_p4 = GatedFusion(768)
            self.fuse_p5 = GatedFusion(1536)
        # mAP50 = 83.6 % 46.84 hour
        elif fusion_type == "cross_attn":
            print("Cross Attention")
            # dropout reduced 0.3->0.1: 0.3 spatial dropout pre-attention was too aggressive
            # P3 fuse: attention will run on pooled (26x26=676 tokens) in forward() to save memory
            self.fuse_p3 = CrossAttnFusion2D(768, 512, 768, num_heads=8, dropout=0.1)
            self.fuse_p4 = CrossAttnFusion2D(768, 512, 768, num_heads=8, dropout=0.1)
            self.fuse_p5 = CrossAttnFusion2D(1536, 1024, 1536, num_heads=8, dropout=0.1)
        # mAP50 = 82.5 %  27.08 hour
        elif fusion_type == "gated_bilinear":
            # LearnableGatedBilinearFusion can handle different input channels
            self.fuse_p3 = LearnableGatedBilinearFusion(768, 512, 768)    # cam=768, lidar=512, out=768
            self.fuse_p4 = LearnableGatedBilinearFusion(768, 512, 768)    # cam=768, lidar=512, out=768
            self.fuse_p5 = LearnableGatedBilinearFusion(1536, 1024, 1536) # cam=1536, lidar=1024, out=1536
        # mAP50 = 82.2 %   27.00 hour
        elif fusion_type == "adapt_gated":
            # AdaptiveWeightedFusion can handle different input channels
            self.fuse_p3 = AdaptiveWeightedFusion(768, 512, 768)    # cam=768, lidar=512, out=768
            self.fuse_p4 = AdaptiveWeightedFusion(768, 512, 768)    # cam=768, lidar=512, out=768
            self.fuse_p5 = AdaptiveWeightedFusion(1536, 1024, 1536) # cam=1536, lidar=1024, out=1536
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # Hybrid head expects camera-sized channels: 768, 768, 1536
        self.head = YOLO11HybridHeadFused(nc=nc, cam_p3_ch=768, cam_p4_ch=768, cam_p5_ch=1536)

    def forward(self, x_cam, x_lidar):
        import torch.nn.functional as F

        p3c, p4c, p5c = self.backbone_cam(x_cam)     # YOLOv11m: 768, 768, 1536
        p3l, p4l, p5l = self.backbone_lidar(x_lidar)  # YOLOv11n: 512, 512, 1024

        # For gated fusion, project LiDAR to match camera channels
        if self.fusion_type == "gated":
            p3l = self.lidar_proj_p3(p3l)  # 512 -> 768
            p4l = self.lidar_proj_p4(p4l)  # 512 -> 768
            p5l = self.lidar_proj_p5(p5l)  # 1024 -> 1536

        if self.fusion_type == "cross_attn":
            # P3 pooled attention: 52x52 -> 26x26 (676 tokens vs 2704 = 4x cheaper)
            # This is the main OOM fix — P3 attention dominates memory at higher resolutions.
            p3_fused = self.fuse_p3(
                F.avg_pool2d(p3c, 2),
                F.avg_pool2d(p3l, 2),
            )
            p3 = F.interpolate(p3_fused, size=p3c.shape[-2:], mode='bilinear', align_corners=False)
            p4 = self.fuse_p4(p4c, p4l)
            p5 = self.fuse_p5(p5c, p5l)
        else:
            p3 = self.fuse_p3(p3c, p3l)
            p4 = self.fuse_p4(p4c, p4l)
            p5 = self.fuse_p5(p5c, p5l)

        return self.head(p3, p4, p5)
    

if __name__ == "__main__":
    model = MultiModalYOLO11(
        nc=10,
        cam_in_ch=3,
        lidar_in_ch=3,
        fusion_type="cross_attn",  # or "gated", "cross_attn",  "gated_bilinear", adapt_gated
    )
    x_cam = torch.randn(2, 3, 640, 640)
    x_lidar = torch.randn(2, 3, 640, 640)
    out = model(x_cam, x_lidar)
    print(type(out), len(out))
    for i, o in enumerate(out):
        print(i, o.shape)


