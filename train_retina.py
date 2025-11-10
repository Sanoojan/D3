import os
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import datetime
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import cv2

from data import D3_dataset_AP
from models import D3_model
from models.D3_model import TransformerClassifier, LocalTemporalClassifier
from Retina_patches.general_utils.config_utils import load_config
from Retina_patches.src.models import get_model
from Retina_patches.src.aligners import get_aligner
from einops import rearrange, reduce, repeat

# =======================
# Utility
# =======================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_batch(batch_input, ldmks=None, foreground_masks=None, show=False, save_path=None):
    """
    Visualizes a batch of video frames with optional landmarks and foreground masks.
    Each landmark index is drawn in a distinct color.

    Args:
        batch_input (torch.Tensor): Tensor of shape (T, 3, H, W) representing video frames.
        ldmks (torch.Tensor or None): Tensor of shape (T, N, 2), normalized landmark coordinates in [0, 1].
        foreground_masks (torch.Tensor or None): Tensor of shape (T, H, W).
        show (bool): If True, displays the plot.
        save_path (str): Path to save the visualization.
    """
    # Convert to CPU numpy
    frames = batch_input.detach().cpu().permute(0, 2, 3, 1).numpy()  # (T, H, W, C)
    frames = np.clip((frames - frames.min()) / (frames.max() - frames.min() + 1e-8), 0, 1)

    T, H, W, _ = frames.shape
    ncols = min(T, 8)
    nrows = int(np.ceil(T / ncols))

    plt.figure(figsize=(3 * ncols, 3 * nrows))

    for i in range(T):
        plt.subplot(nrows, ncols, i + 1)
        frame = frames[i]
        plt.imshow(frame)

        # Overlay mask
        if foreground_masks is not None:
            mask = foreground_masks[i].detach().cpu().numpy()
            plt.imshow(mask, cmap='Reds', alpha=0.3)

        # Overlay landmarks
        if ldmks is not None:
            landmarks = ldmks[i].detach().cpu().numpy()
            valid = (landmarks[:, 0] >= 0) & (landmarks[:, 1] >= 0)
            landmarks = landmarks[valid]

            if len(landmarks) > 0:
                num_ldmks = landmarks.shape[0]
                cmap = plt.cm.get_cmap('rainbow', num_ldmks)
                x = landmarks[:, 0] * W
                y = landmarks[:, 1] * H
                for j in range(num_ldmks):
                    plt.scatter(x[j], y[j], s=25, color=cmap(j), edgecolors='black', linewidths=0.5)

        plt.axis('off')

    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Visualization saved at {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

# =======================
# Main
# =======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train script for D3 model with separate train/test CSVs.')
    
    # General args
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=str, default="1")
    parser.add_argument('--loss', type=str, default='cos', choices=['l2', 'cos'])
    parser.add_argument('--encoder', type=str, default='XCLIP-16')
    parser.add_argument('--ckpt-dir', type=str, default="Retina_patches/checkpoints/sapiensid_wb4m")
    parser.add_argument('--aligner', type=str, default='yolo_dfa.yaml')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--classifier-type', type=str, default='Transformer', choices=['Transformer', 'local_temporal'])

    # Train/Test CSVs
    parser.add_argument('--train-real-csv', type=str, default="dataset/DeepfakeDatasets/FakeAVCeleb/csv/train_real.csv")
    parser.add_argument('--train-fake-csv', type=str, default="dataset/DeepfakeDatasets/FakeAVCeleb/csv/train_fake.csv")
    parser.add_argument('--test-real-csv', type=str, default="dataset/DeepfakeDatasets/FakeAVCeleb/csv/test_real.csv")
    parser.add_argument('--test-fake-csv', type=str, default="dataset/DeepfakeDatasets/FakeAVCeleb/csv/test_fake.csv")

    args = parser.parse_args()

    # =======================
    # Setup
    # =======================
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    print(f"\nTraining {args.encoder} model with {args.loss} loss")
    os.makedirs(args.save_dir, exist_ok=True)

    # =======================
    # Load model + aligner
    # =======================
    model_config = load_config(os.path.join(args.ckpt_dir, 'model.yaml'))
    model = get_model(model_config).to(device)
    
    if args.classifier_type == 'Transformer':

        classifier_model=TransformerClassifier(in_dim=512).to(device)
        
    elif args.classifier_type == 'local_temporal':
        classifier_model=LocalTemporalClassifier(num_patches=252).to(device)

    if args.aligner.endswith('.yaml'):
        aligner_config = load_config(os.path.join('Retina_patches/src/aligners/configs', args.aligner))
        if hasattr(model_config, 'rgb_mean'):
            aligner_config.rgb_mean = model_config.rgb_mean
            aligner_config.rgb_std = model_config.rgb_std
        aligner = get_aligner(aligner_config)
    aligner = aligner.to(device)

    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # =======================
    # Datasets
    # =======================
    train_dataset = D3_dataset_AP(real_csv=args.train_real_csv, fake_csv=args.train_fake_csv, max_len=1000, aug_type='custom_series')
    test_dataset = D3_dataset_AP(real_csv=args.test_real_csv, fake_csv=args.test_fake_csv, max_len=1000)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # =======================
    # Training loop
    # =======================
    best_auc = 0.0

    pos_weight=torch.tensor(train_dataset.num_fake_samples / train_dataset.num_real_samples)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in range(args.epochs):
        model.eval()
        classifier_model.train()
        running_loss = 0.0
        for batch_frames, batch_label in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]"):
            batch_inputs = batch_frames.to(device) #B,T,C,H,W
            batch_label = batch_label.to(device) #B
            features_all = []
            with torch.no_grad():
                for i in range(batch_inputs.size(0)):
                    # visualize_batch(batch_inputs[i])
                    features = aligner(batch_inputs[i:i+1])
                    features_all.append(features)
            features = features_all
            # visualize_batch(batch_inputs[0], ldmks=features[0][0], foreground_masks=features[0][1], show=False, save_path=f"visualizations/epoch{epoch+1}_batch_sample.png")

            B,T,C,H,W=batch_inputs.size()
            batch_inputs = rearrange(batch_inputs, 'b t c h w -> (b t) c h w')
            ldmks = torch.cat([f[0] for f in features], dim=0)
            foreground_masks = torch.cat([f[1] for f in features], dim=0)
            outputs = model(batch_inputs, ldmks=ldmks, foreground_masks=foreground_masks)

            if args.classifier_type == 'Transformer':
                outputs=outputs[:,0,:]
                outputs = rearrange(outputs, '(b t) d-> b t d', b=B, t=T)
                outputs = classifier_model(outputs)
            elif args.classifier_type == 'local_temporal':
                outputs=outputs[:, 1: , :]
                outputs = rearrange(outputs, '(b t) p d-> b t p d', b=B, t=T)
                outputs = classifier_model(outputs) 
            # vec1 = outputs[:, :-1, :]
            # vec2 = outputs[:, 1:, :]

            # if args.loss == 'cos':
            #     loss = 1 - F.cosine_similarity(vec1, vec2, dim=-1).mean()
            # else:
            #     loss = torch.norm(vec1 - vec2, p=2, dim=-1).mean()
            loss = criterion(outputs, batch_label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # =======================
        # Test Evaluation
        # =======================
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch_frames, batch_label in tqdm(test_loader, desc="Evaluating on Test Set"):
                    batch_inputs = batch_frames.to(device)
                    
                    features_all = []
                    
                    for i in range(batch_inputs.size(0)):
                        # visualize_batch(batch_inputs[i])
                        features = aligner(batch_inputs[i:i+1])
                        features_all.append(features)
                    features = features_all
                    B,T,C,H,W=batch_inputs.size()
                    batch_inputs = rearrange(batch_inputs, 'b t c h w -> (b t) c h w')
                    ldmks = torch.cat([f[0] for f in features], dim=0)
                    foreground_masks = torch.cat([f[1] for f in features], dim=0)
                    outputs = model(batch_inputs, ldmks=ldmks, foreground_masks=foreground_masks)
                    
                    if args.classifier_type == 'Transformer':
                        outputs=outputs[:,0,:]
                        outputs = rearrange(outputs, '(b t) d-> b t d', b=B, t=T)
                        outputs = classifier_model(outputs)
                    elif args.classifier_type == 'local_temporal':
                        outputs=outputs[:, 1: , :]
                        outputs = rearrange(outputs, '(b t) p d-> b t p d', b=B, t=T)
                        outputs = classifier_model(outputs) 
                    
                    # vec1 = outputs[:, :-1, :]
                    # vec2 = outputs[:, 1:, :]
                    # dis_1st = F.cosine_similarity(vec1, vec2, dim=-1)
                    # dis_2nd = dis_1st[:, 1:] - dis_1st[:, :-1]
                    # dis_std = torch.std(dis_2nd, dim=1)
                    # final_out=dis_std

                    final_out=outputs.squeeze()
                    y_pred.extend(final_out.cpu().numpy())
                    y_true.extend(batch_label.cpu().numpy())

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            ap = average_precision_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            print(f"Test - AP: {ap:.4f} | AUC: {auc:.4f}")

            #save all models
            ckpt_path = os.path.join(args.save_dir, f"model_epoch{epoch+1}.pth")
            torch.save(classifier_model.state_dict(), ckpt_path)
            print(f"✅ Saved model to {ckpt_path}")

            # Save best model
            if auc > best_auc:
                best_auc = auc
                ckpt_path = os.path.join(args.save_dir, f"best_model_epoch{epoch+1}.pth")
                torch.save(classifier_model.state_dict(), ckpt_path)
                print(f"✅ Saved best model to {ckpt_path}")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")