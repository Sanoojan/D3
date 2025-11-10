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

def get_distinct_colors(n):
    cmap = plt.get_cmap('tab20')
    return [cmap(i) for i in range(n)]

def denormalize_image(img, mean, std, max_pixel_value=255.0):
    img = img.copy()
    mean_arr = np.array(mean).reshape(3, 1, 1)
    std_arr = np.array(std).reshape(3, 1, 1)
    img = img * (std_arr * max_pixel_value) + (mean_arr * max_pixel_value)
    img = np.clip(img, 0, max_pixel_value).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    return img

def visualize_patches_and_landmarks(
    image,
    patch_locations,
    landmarks,
    save_path=None,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_pixel_value=255.0
):
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()
    image = denormalize_image(image, mean, std, max_pixel_value)
    H, W = image.shape[:2]
    valid_indices = [i for i, lm in enumerate(landmarks) if not (lm < -0.9999).all()]
    colors = get_distinct_colors(len(valid_indices))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, interpolation='nearest')

    # Draw landmarks
    for color_idx, lm_idx in enumerate(valid_indices):
        lm = landmarks[lm_idx]
        px = int(lm[0] * W)
        py = int(lm[1] * H)
        ax.scatter(px, py, c=[colors[color_idx]], s=80, marker='o', label=f'LM {lm_idx}')
    
    # Draw patches
    for patch in patch_locations:
        x, y, w, h, landmark_id = patch
        if (landmark_id is not None) and (landmark_id in valid_indices):
            color = colors[valid_indices.index(landmark_id)]
        else:
            color = "blue"
        rect = patches.Rectangle((x, y), w, h,
                                linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
        ax.add_patch(rect)

    ax.set_title("Patches (colored/fine/coarse) & Landmarks")
    ax.axis('off')
    if len(valid_indices) > 0: plt.legend()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

class CustomPatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels,
                 embed_dim,
                 fine_patch=16,
                 coarse_patch=64):
        super().__init__()
        self.fine_patch = fine_patch
        self.coarse_patch = coarse_patch
        # Store projection layers for both patch sizes (or share if you want)
        self.proj_fine = nn.Linear(in_channels * fine_patch * fine_patch, embed_dim)
        self.proj_coarse = nn.Linear(in_channels * coarse_patch * coarse_patch, embed_dim)

    def forward(self, images, patch_locations):
        """
        images: Tensor (B, C, H, W)
        patch_locations: list for each image, each inner list is (x, y, w, h, landmark_id)
        returns:
            patch_embeddings: (B, N, embed_dim) tensor (where N varies by image if using list of tensors)
        """
        B, C, H, W = images.shape
        all_embeddings = []
        for i in range(B):
            img = images[i]
            emb_list = []
            for (x, y, w, h, landmark_id) in patch_locations[i]:
                patch = img[:, y:y+h, x:x+w]  # (C, h, w)
                patch_flat = patch.reshape(-1)  # (C*h*w,)
                if h == self.fine_patch and w == self.fine_patch:
                    emb = self.proj_fine(patch_flat)
                elif h == self.coarse_patch and w == self.coarse_patch:
                    emb = self.proj_coarse(patch_flat)
                else:
                    # For any other size, interpolate to fine_patch or coarse_patch then embed
                    patch_resized = F.interpolate(patch.unsqueeze(0), 
                                                  size=(self.fine_patch, self.fine_patch) if h<w else (self.coarse_patch, self.coarse_patch),
                                                  mode='bilinear', align_corners=False).squeeze(0)
                    patch_flat = patch_resized.reshape(-1)
                    emb = self.proj_fine(patch_flat) if h<w else self.proj_coarse(patch_flat)
                emb_list.append(emb)
            all_embeddings.append(torch.stack(emb_list, dim=0))  # (num_patches, embed_dim)
        # Optionally, pad or pack resulting embeddings for batching (B, max_num_patches, embed_dim)
        max_num_patches = max([embs.shape[0] for embs in all_embeddings])
        batch_embeddings = []
        for embs in all_embeddings:
            if embs.shape[0] < max_num_patches:
                pad = torch.zeros((max_num_patches - embs.shape[0], embs.shape[1]), device=embs.device, dtype=embs.dtype)
                embs = torch.cat([embs, pad], dim=0)
            batch_embeddings.append(embs)
        patch_embeddings = torch.stack(batch_embeddings, dim=0)
        return patch_embeddings  # (B, max_num_patches, embed_dim)

def custom_patchification(images, landmarks, fine_patch=16, coarse_patch=64, fine_radius=32):
    """
    images: Tensor (B, 3, 224, 224)
    landmarks: Tensor (B, 22, 2), [0,1] normalized; invalid is (-1,-1)
    Returns:
        patches: list of Tensor lists for each image
        patch_locations: list of (x, y, w, h, landmark_id) per patch for each image
    """
    B, C, H, W = images.shape
    patches = []
    patch_locations = []
    for b in range(B):
        img = images[b]
        lms = landmarks[b]
        used_mask = torch.zeros(H, W, dtype=torch.bool, device=img.device)
        all_patches, all_locs = [], []
        # Finer patches near each valid landmark
        for landmark_id, lm in enumerate(lms):
            if (lm < -0.9999).all(): continue
            center_x = int(lm[0] * W)
            center_y = int(lm[1] * H)
            if not (0 <= center_x < W and 0 <= center_y < H): continue
            for dx in range(-fine_radius, fine_radius + 1, fine_patch):
                for dy in range(-fine_radius, fine_radius + 1, fine_patch):
                    x = center_x + dx - fine_patch // 2
                    y = center_y + dy - fine_patch // 2
                    if x < 0 or y < 0 or x + fine_patch > W or y + fine_patch > H:
                        continue
                    if used_mask[y:y+fine_patch, x:x+fine_patch].any():
                        continue
                    patch = img[:, y:y+fine_patch, x:x+fine_patch]
                    all_patches.append(patch)
                    all_locs.append((x, y, fine_patch, fine_patch, landmark_id))
                    used_mask[y:y+fine_patch, x:x+fine_patch] = True
        # Coarse patches elsewhere
        for y in range(0, H, coarse_patch):
            for x in range(0, W, coarse_patch):
                if used_mask[y:y+coarse_patch, x:x+coarse_patch].any():
                    continue
                if x + coarse_patch > W or y + coarse_patch > H:
                    continue
                patch = img[:, y:y+coarse_patch, x:x+coarse_patch]
                all_patches.append(patch)
                all_locs.append((x, y, coarse_patch, coarse_patch, None))
                used_mask[y:y+coarse_patch, x:x+coarse_patch] = True
        patches.append(all_patches)
        patch_locations.append(all_locs)
    return patches, patch_locations


def normalized_to_pixel(ldmks, H, W):
    # mask invalid (-1,-1)
    valid = (ldmks[:,0] >= 0) & (ldmks[:,1] >= 0)
    ldmks_pix = ldmks.clone()

    ldmks_pix[valid, 0] = ldmks_pix[valid, 0] * W  # x * width
    ldmks_pix[valid, 1] = ldmks_pix[valid, 1] * H  # y * height

    # set invalid to large far-away coordinate so they don't affect distance
    ldmks_pix[~valid] = 1e9
    return ldmks_pix
def visualize_importance_patches_with_ldmks(
    img,
    importance,
    ldmks,
    num_patches=196,
    patch_size=16,
    save="adaptive_overlay_ldmks.png"
):
    H, W = importance.shape
    assert img.shape[1] == H and img.shape[2] == W

    # Normalize image
    img = (img - img.min())/(img.max()-img.min())
    img_pil = torchvision.transforms.functional.to_pil_image(img.cpu())
    draw = ImageDraw.Draw(img_pil)

    #### ---- Draw patches ---- ####
    prob = importance.flatten()
    prob = prob / prob.sum()

    idx = torch.multinomial(prob, num_patches, replacement=True)
    ys = (idx // W).cpu().numpy()
    xs = (idx % W).cpu().numpy()

    half = patch_size // 2
    for x, y in zip(xs, ys):
        x1 = max(0, x-half)
        x2 = min(W, x+half)
        y1 = max(0, y-half)
        y2 = min(H, y+half)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    #### ---- Draw landmarks ---- ####
    # valid: normalized ldmks in [0,1]
    valid = (ldmks[:,0] >= 0) & (ldmks[:,1] >= 0)
    ldmks_valid = ldmks[valid]

    # Predefined distinct colors (you can add more)
    palette = [
        "blue", "green", "yellow", "cyan", "magenta", "white",
        "orange", "purple", "lime", "pink", "gold"
    ]
    colors = [palette[i % len(palette)] for i in range(len(ldmks_valid))]

    # Convert normalized -> pixel coords
    xs = (ldmks_valid[:,0] * W).cpu().numpy().astype(int)
    ys = (ldmks_valid[:,1] * H).cpu().numpy().astype(int)

    # Draw points
    r = 3  # dot radius
    for (x, y), c in zip(zip(xs, ys), colors):
        draw.ellipse([x-r, y-r, x+r, y+r], outline=c, fill=c)

    img_pil.save("visualization_" + save)
    print(f"✅ Saved adaptive patch + landmark visualization → {save}")

def visualize_importance_patches(img, importance, num_patches=196, patch_size=16, save="adaptive_overlay.png"):
    H, W = importance.shape
    assert img.shape[1] == H and img.shape[2] == W

    img = (img - img.min())/(img.max()-img.min())
    img_pil = torchvision.transforms.functional.to_pil_image(img.cpu())
    draw = ImageDraw.Draw(img_pil)

    # flatten importance to sample indices
    prob = importance.flatten()
    prob = prob / prob.sum()

    idx = torch.multinomial(prob, num_patches, replacement=True)
    ys = (idx // W).cpu().numpy()
    xs = (idx % W).cpu().numpy()

    half = patch_size // 2
    for x, y in zip(xs, ys):
        x1 = max(0, x-half)
        x2 = min(W, x+half)
        y1 = max(0, y-half)
        y2 = min(H, y+half)
        draw.rectangle([x1,y1,x2,y2], outline="red", width=2)

    img_pil.save(save)
    print(f"✅ Saved adaptive patch visualization → {save}")
    
def visualize_adaptive_patches(pixel_values, landmarks, patch_radius=16,
                               save_dir="adaptive_patches", prefix="sample"):
    """
    pixel_values: (B, 3, H, W) input image tensor 
    landmarks: (B, N, 2) pixel coordinates (x, y)
    patch_radius: half patch size (e.g. 16 -> 32x32 patches)
    """

    os.makedirs(save_dir, exist_ok=True)

    # Convert tensor to image
    img = pixel_values[0].detach().cpu()
    img = (img - img.min()) / (img.max() - img.min())
    img_pil = TF.to_pil_image(img)
    draw = ImageDraw.Draw(img_pil)

    B, C, H, W = pixel_values.shape
    lm = torch.round(landmarks[0]).cpu().int()  # first sample landmarks

    patch_count = 0
    for (x, y) in lm:
        x, y = int(x), int(y)

        # Patch bounds
        x1 = max(0, x - patch_radius)
        x2 = min(W, x + patch_radius)
        y1 = max(0, y - patch_radius)
        y2 = min(H, y + patch_radius)

        # Draw patch box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Extract patch
        patch = pixel_values[0, :, y1:y2, x1:x2]
        patch = (patch - patch.min()) / (patch.max() - patch.min())
        patch_img = TF.to_pil_image(patch.cpu())

        patch_img.save(f"{save_dir}/{prefix}_patch_{patch_count:03d}.png")
        patch_count += 1

    # Save overlay
    img_pil.save(f"{save_dir}/{prefix}_overlay.png")

    print(f"✅ Saved {patch_count} adaptive patches + overlay to {save_dir}/")

def visualize_xclip_patches(pixel_values, save_dir="patch_vis", prefix="sample"):
    """
    pixel_values: tensor of shape (B,3,224,224)
    save_dir: folder to save patches and overlay image
    """
    os.makedirs(save_dir, exist_ok=True)

    B, C, H, W = pixel_values.shape
    assert H == 224 and W == 224, "Expected input resolution 224×224"
    patch_size = 16
    grid = 14  # 224/16

    # Normalize tensor to 0-255 for saving as images
    img = pixel_values[0].detach().cpu()
    img = (img - img.min()) / (img.max() - img.min())
    img = TF.to_pil_image(img)

    # --- Draw patches on original image ---
    draw = ImageDraw.Draw(img)
    patch_count = 0

    for row in range(grid):
        for col in range(grid):
            y1, y2 = row * patch_size, (row + 1) * patch_size
            x1, x2 = col * patch_size, (col + 1) * patch_size

            # draw box
            draw.rectangle([x1, y1, x2, y2], outline="red")

            # extract patch and save
            patch = pixel_values[0, :, y1:y2, x1:x2]
            patch = (patch - patch.min()) / (patch.max() - patch.min())
            patch_img = TF.to_pil_image(patch.cpu())
            patch_img.save(f"{save_dir}/{prefix}_patch_{patch_count:03d}.jpg")

            patch_count += 1

    # save overlay
    img.save(f"{save_dir}/{prefix}_patch_overlay.jpg")
    print(f"✅ Saved {patch_count} patches + overlay to {save_dir}/")