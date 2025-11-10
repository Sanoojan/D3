
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from transformers import CLIPVisionModel, XCLIPVisionModel, AutoModel
from transformers.models.x_clip.modeling_x_clip import XCLIPVisionEmbeddings
import torchvision.models as models
import os
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
import torchvision
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from models.custom_patch import *

Transformers = [
    'CLIP-16',
    'CLIP-32',
    'XCLIP-16',
    'XCLIP-32',
    'DINO-base',
    'DINO-large',
]




class D3_model(nn.Module):
    def __init__(self, encoder_type = 'CLIP-16', loss_type = 'cos'):
        super(D3_model, self).__init__()
        self.loss_type = loss_type
        self.encoder_type = encoder_type
        

        if encoder_type == 'CLIP-16':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

        elif encoder_type == 'CLIP-32':
            self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        elif encoder_type == 'XCLIP-16':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16")
            self.encoder.vision_model.embeddings = customXCLIPVisionEmbeddings(self.encoder.vision_model.config)

        elif encoder_type == 'XCLIP-32':
            self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")

        elif encoder_type == 'DINO-base':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-base")

        elif encoder_type == 'DINO-large':
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-large")

        elif encoder_type == 'ResNet-18':
            resnet18 = models.resnet18(pretrained=True)
            modules = list(resnet18.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules).eval()

        elif encoder_type == 'VGG-16':
            vgg16 = models.vgg16(pretrained=True)
            modules = list(vgg16.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules).eval()

        elif encoder_type == 'EfficientNet-b4':
            efficientnet_b4 = models.efficientnet_b4(pretrained=True)
            modules = list(efficientnet_b4.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules).eval() 

        elif encoder_type == 'MobileNet-v3':
            mobilenetv3 = timm.create_model('mobilenetv3_large_100', pretrained=True)
            modules = list(mobilenetv3.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules).eval() 

    def encoder_forward(self, x):
        b, t, _, h, w = x.shape
        images = x.reshape(-1, 3, h, w)
        if self.encoder_type in Transformers:
            outputs = self.encoder(images, output_hidden_states=True)
            outputs = outputs.pooler_output
        else:
            outputs = self.encoder(images)
        outputs=outputs.reshape(b, t, -1)
        return outputs
    
    def forward_custom_patch(self, x, ldmks=None):
        self.encoder.vision_model.embeddings.ldmks=ldmks
        # self.encoder.vision_model.embeddings.foreground_masks=foreground_masks
        b, t, _, h, w = x.shape
        images = x.reshape(-1, 3, h, w)
        if self.encoder_type in Transformers:
            outputs = self.encoder(images, output_hidden_states=True,ldmks=ldmks)
            # outputs = outputs.last_hidden_state
            outputs = outputs.pooler_output
        else:
            outputs = self.encoder(images,ldmks=ldmks)
            
        # outputs=outputs[:, 1:, :] # remove class token
        # _,N,D=outputs.shape
        
        # outputs=outputs.reshape(b, t, N, D)
        outputs=outputs.reshape(b, t, -1)
        vec1 = outputs[:, :-1, :]  # [b, n-1, 768]
        vec2 = outputs[:, 1:, :]   # [b, n-1, 768]
        if self.loss_type == 'cos':
            dis_1st = F.cosine_similarity(vec1, vec2, dim=-1)  # [b, n-1]
        elif self.loss_type == 'l2':
            dis_1st = torch.norm(vec1 - vec2, p=2, dim=-1)  # [b, n-1]
        # dis_2nd = dis_1st[:, 1:] - dis_1st[:, :-1]  # [b, n-2]
        # dis_2nd_avg = torch.mean(dis_2nd,dim=1)
        # dis_2nd_std = torch.std(dis_2nd, dim=1) # [b]
        return outputs # [B, T-1]
    
    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.reshape(-1, 3, h, w)
        if self.encoder_type in Transformers:
            outputs = self.encoder(images, output_hidden_states=True)
            outputs = outputs.pooler_output
        else:
            outputs = self.encoder(images)
        outputs=outputs.reshape(b, t, -1)
        vec1 = outputs[:, :-1, :]  # [b, n-1, 768]
        vec2 = outputs[:, 1:, :]   # [b, n-1, 768]
        if self.loss_type == 'cos':
            dis_1st = F.cosine_similarity(vec1, vec2, dim=-1)  # [b, n-1]
        elif self.loss_type == 'l2':
            dis_1st = torch.norm(vec1 - vec2, p=2, dim=-1)  # [b, n-1]
        dis_2nd = dis_1st[:, 1:] - dis_1st[:, :-1]  # [b, n-2]
        dis_2nd_avg = torch.mean(dis_2nd,dim=1)
        dis_2nd_std = torch.std(dis_2nd, dim=1) # [b]
        return outputs, dis_2nd_avg, dis_2nd_std
    
class TransformerClassifier(nn.Module):
    def __init__(self, in_dim, num_heads=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.transformer(x)   # (B, T, D)
        x = x.mean(dim=1)         # global average pooling
        logits = self.fc(x)
        return logits.squeeze(1)

class CosineTemporalClassifier(nn.Module):
    def __init__(self, num_patches, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Project per-patch similarities into a higher-dimensional embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(num_patches, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Temporal modeling (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head (video-level)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        x: (B, T, N) - cosine similarity maps across frames and patches
        """
        # Step 1: Patch embedding
        x = self.patch_embed(x)    # (B, T, embed_dim)
        
        # Step 2: Temporal encoding
        x = self.transformer(x)    # (B, T, embed_dim)
        
        # Step 3: Temporal pooling (global average)
        x = x.mean(dim=1)          # (B, embed_dim)
        
        # Step 4: Classification
        logits = self.fc(x).squeeze(1)   # (B,)
        return logits
    
class LocalTemporalClassifier(nn.Module):
    def __init__(self, num_patches, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_patches, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x_t = x[:, :-1, :, :]
        x_tp1 = x[:, 1:, :, :]
        diff = (x_tp1 - x_t).pow(2).sum(dim=-1).sqrt()  # [B, T-1, P]
        motion = diff.mean(dim=1)  # [B, P]
        logits = self.fc(motion)
        return logits.squeeze(-1)

def compute_importance_map(h, w, ldmks, tau=50):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).float().to(ldmks.device)  # [h,w,2]

    dists = torch.cdist(coords.reshape(-1,2), ldmks.float())  # [hw, N]
    min_dist = dists.min(dim=1)[0].reshape(h,w)
    importance = torch.exp(-min_dist / tau)
    importance /= importance.sum()
    return importance


def sample_patch_centers(importance, num_patches=196):
    flat = importance.flatten()
    idx = torch.multinomial(flat, num_patches, replacement=True)
    h, w = importance.shape
    y = idx // w
    x = idx % w
    return torch.stack([y, x], dim=1)  # [196,2]


def adaptive_patch_size(dist, min_patch=8, max_patch=32):
    closeness = torch.exp(-dist / 50)
    return (max_patch - closeness * (max_patch - min_patch)).long().clamp(min_patch, max_patch)


def extract_patch(img, cy, cx, p):
    y1 = max(0, cy - p//2)
    y2 = min(img.shape[1], cy + p//2)
    x1 = max(0, cx - p//2)
    x2 = min(img.shape[2], cx + p//2)

    patch = img[:, y1:y2, x1:x2]   # [3,H,W]
    patch = F.interpolate(patch.unsqueeze(0), size=(16,16), mode='bilinear')
    return patch.squeeze(0)  # [3,16,16]


class customXCLIPVisionEmbeddings(XCLIPVisionEmbeddings):
    """
    Adaptive patching XCLIP embedding:
    - 196 adaptive patches based on facial landmarks
    - preserves class token & position embeddings
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Replace Conv2D patch embed with linear embedding
        self.adaptive_proj = nn.Linear(16*16*3, self.embed_dim)
        self.num_patches = (config.image_size // config.patch_size) ** 2  # 196
        self.custom_patch_embedding= CustomPatchEmbedding(
            in_channels=3,      # for RGB images
            embed_dim=768,      # size of your patch embedding (should match transformer)
            fine_patch=16,      # size of fine patches (default)
            coarse_patch=64     # size of coarse patches (default)
        )

    def adaptive_patch_embed(self, pixel_values, ldmks):
        b, c, h, w = pixel_values.shape
        all_patches = []

        for i in range(b):
            img = pixel_values[i]
            ldmks_pix= normalized_to_pixel(ldmks[i], h, w)
            imp = compute_importance_map(h, w, ldmks_pix)
            centers = sample_patch_centers(imp, self.num_patches)
            visualize_importance_patches_with_ldmks(
                img,
                imp,
                ldmks[i],
                num_patches=196,
                patch_size=16,
                save="adaptive_overlay_ldmks.png"
            )
        
            dists = torch.cdist(centers.float(), ldmks_pix.float()).min(dim=1)[0]
            patch_sizes = adaptive_patch_size(dists)

            patches = []
            for (cy, cx), p in zip(centers, patch_sizes):
                patches.append(extract_patch(img, cy.item(), cx.item(), p.item()).flatten())
            
            patches = torch.stack(patches)  # [196, 3*16*16]
            patches = self.adaptive_proj(patches)  # [196, embed_dim]
            all_patches.append(patches)

        return torch.stack(all_patches, dim=0)  # [B,196,embed_dim]

    def forward(self, pixel_values, ldmks, foreground_masks=None, interpolate_pos_encoding=False):
        b, _, h, w = pixel_values.shape

        if (h != self.image_size) or (w != self.image_size):
            raise ValueError(f"Input size {h}×{w} must match model {self.image_size}×{self.image_size}")

        # Adaptive patch embeddings
        # patch_embeds = self.adaptive_patch_embed(pixel_values, ldmks)
        patch_embeds, patch_locations = custom_patchification(pixel_values, ldmks)
        # visualize_patches_and_landmarks(pixel_values[0], patch_locations[0], ldmks[0], save_path="visualizations/patches_landmarks1.png")

        patch_embeds= self.custom_patch_embedding(pixel_values, patch_locations)
        
        # Class token
        class_embeds = self.class_embedding.expand(b, 1, -1)

        # [B, 1 + 196, embed_dim]
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        self.position_ids = torch.arange(embeddings.size(1), device=embeddings.device).unsqueeze(0)
        # Positional embed
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, h, w)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings