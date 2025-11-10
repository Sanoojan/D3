import os
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import datetime
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from data import D3_dataset_AP
from models import D3_model
from Retina_patches.general_utils.config_utils import load_config
from Retina_patches.src.models import get_model
from Retina_patches.src.aligners import get_aligner
from Retina_patches.src.evaluations import get_evaluator_by_name, summary
from Retina_patches.src.pipelines import pipeline_from_name
from Retina_patches.src.fabric.fabric import setup_dataloader_from_dataset
import torch.nn.functional as F

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script with configurable parameters.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--gpu-id', type=str, default="0",
                        help='CUDA GPU device ID(s), e.g., "0" or "1,2,3" (default: "5")')
    parser.add_argument('--loss', type=str, default='cos', choices=['l2', 'cos'],
                        help='Loss function type (default: l2)')
    parser.add_argument('--encoder', type=str, default='XCLIP-16', 
                        help='Encoder model name (default: XCLIP-16)',
                        choices=['CLIP-16', 'CLIP-32', 'XCLIP-16', 'XCLIP-32', 'DINO-base', 'DINO-large', 'ResNet-18', 'VGG-16', 'EfficientNet-b4', 'MobileNet-v3'])
    parser.add_argument('--real-csv', type=str, default="dataset/DeepfakeDatasets/FakeAVCeleb/csv/RealVideo-RealAudio.csv",
                        help='Path to the real data CSV file ')
    parser.add_argument('--fake-csv', type=str, default="dataset/DeepfakeDatasets/FakeAVCeleb/csv/FakeVideo-RealAudio.csv",
                        help='Path to the fake/synthetic data CSV file')
    parser.add_argument('--ckpt-dir', type=str,default="Retina_patches/checkpoints/sapiensid_wb4m",
                        help='Path to the directory containing the model checkpoint and config')
    parser.add_argument('--aligner', type=str, default='yolo_dfa.yaml')
    args = parser.parse_args()

    seed = args.seed
    gpu_id = args.gpu_id
    loss_type = args.loss
    encoder_type = args.encoder
    real_csv = args.real_csv
    fake_csv = args.fake_csv

    # real_csv = 'datasets/csv/t1.csv'
    # fake_csv = 'datasets/csv/t2.csv' 
    
    print(f"Starting AP evaluation for {encoder_type} with {loss_type} loss")
    print(f"Real CSV: {real_csv}")
    print(f"Fake CSV: {fake_csv}")
    
    # Load Model
    # model = D3_model(encoder_type=encoder_type, loss_type=loss_type).cuda()
    
    model_config = load_config(os.path.join(args.ckpt_dir, 'model.yaml'))
    model = get_model(model_config)
    model.load_state_dict_from_path(os.path.join(args.ckpt_dir, 'model.pth'))
    model.eval()
    model = model.cuda()
    
    if args.aligner != '':
        if args.aligner.endswith('.yaml'):
            aligner_config = load_config(os.path.join('Retina_patches/src/aligners/configs', args.aligner))
            if hasattr(model_config, 'rgb_mean'):
                aligner_config.rgb_mean = model_config.rgb_mean
                aligner_config.rgb_std = model_config.rgb_std
            aligner = get_aligner(aligner_config)
        else:
            from transformers import AutoModel
            HF_TOKEN = os.environ['HF_TOKEN']
            aligner = AutoModel.from_pretrained(f"minchul/{args.aligner}",
                                              trust_remote_code=True,
                                              token=HF_TOKEN).model
            aligner.has_params = lambda : True
        # output_dir = os.path.join(output_dir + '_' + args.aligner)
        # output_dir = output_dir.replace('.yaml', '')

    # aligner=aligner.cuda()
    # Load Dataset
    eval_dataset = D3_dataset_AP(real_csv=real_csv, fake_csv=fake_csv, max_len=1000)
    print(f"Total samples: {len(eval_dataset)}")
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True,
        drop_last=False
    )
    compute_second_order = True
    loss_type= args.loss
    # Eval
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_frames, batch_label in tqdm(eval_loader, desc="Evaluating"):
            batch_inputs = batch_frames.cuda()
            
            features=aligner(batch_inputs)
            
            
            # features=features.to("cpu")
            outputs= model(batch_inputs,ldmks=features[0], foreground_masks=features[1])
            print(outputs.shape)
            outputs=outputs.unsqueeze(0)
            if compute_second_order:
                vec1 = outputs[:, :-1, :]  # [b, n-1, 768]
                vec2 = outputs[:, 1:, :]   # [b, n-1, 768]
                if loss_type == 'cos':
                    dis_1st = F.cosine_similarity(vec1, vec2, dim=-1)  # [b, n-1]
                elif loss_type == 'l2':
                    dis_1st = torch.norm(vec1 - vec2, p=2, dim=-1)  # [b, n-1]
                dis_2nd = dis_1st[:, 1:] - dis_1st[:, :-1]  # [b, n-2]
                dis_2nd_avg = torch.mean(dis_2nd,dim=1)
                dis_2nd_std = torch.std(dis_2nd, dim=1) # [b]
                outputs, dis_2nd_avg, dis_2nd_std
            batch_dis_std = dis_2nd_std
            # _, _, batch_dis_std=outputs
            y_pred.extend(batch_dis_std.cpu().flatten().numpy())
            y_true.extend(batch_label.cpu().flatten().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ap_score = average_precision_score(1-y_true, y_pred)
    auc = roc_auc_score(1-y_true, y_pred)
    
    result_str = (
        f"AP Evaluation Results\n"
        f"Encoder: {encoder_type}\n"
        f"Loss Type: {loss_type}\n"
        f"Real CSV: {real_csv}\n"
        f"Fake CSV: {fake_csv}\n"
        f"Real Samples: {np.sum(y_true==0)}\n"
        f"Fake Samples: {np.sum(y_true==1)}\n"
        f"Total Samples: {len(y_true)}\n"
        f"AP Score: {ap_score:.4f}\n"
        f"AUC Score: {auc:.4f}\n"
    )
    
    print("\n" + "="*50)
    print(result_str.strip())
    print("="*50)
    
    os.makedirs("results", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/result_{timestamp}.txt"

    with open(output_file, 'w') as f:
        f.write(result_str)

    print(f"\nResults saved to {output_file}")