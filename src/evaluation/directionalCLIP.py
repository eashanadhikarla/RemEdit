import sys
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from losses.clip_loss import CLIPLoss

##############
## Version 2
##############

def main(original_dir, edited_dir, caption1, caption2):
    original_dir = Path(original_dir)
    edited_dir = Path(edited_dir)

    ## CLIPLoss setup
    clip_loss_fn = CLIPLoss(device="cuda", lambda_direction=1.0)
    clip_loss_fn.set_text_features(caption1, caption2)
    transform = clip_loss_fn.clip_preprocess  # Use the same preprocessing as the model expects

    ## Score lists
    # scores_clip_sim = []
    scores_clip_loss = []

    image_paths = sorted(original_dir.glob("*_original.png"))
    print(f"Total images: {len(image_paths)}")

    for original_path in image_paths:
        edited_path = edited_dir / original_path.name.replace("_original", "_edited")
        if not edited_path.exists():
            print(f"Missing: {edited_path.name}\n{edited_path}\n")
            continue

        ## Load images
        original_image = Image.open(original_path).convert("RGB")
        edited_image = Image.open(edited_path).convert("RGB")

        ## Directional Sdir (image-text) using CLIP loss
        original_tensor = transform(original_image).unsqueeze(0).to("cuda")
        edited_tensor = transform(edited_image).unsqueeze(0).to("cuda")
        # sdir_clip_loss = clip_loss_fn.clip_directional_loss(original_tensor, caption1, edited_tensor, caption2).item() # double check for correctness
        sdir_clip_loss = 1.0 - clip_loss_fn.clip_directional_loss(original_tensor, caption1, edited_tensor, caption2).item()
        scores_clip_loss.append(sdir_clip_loss)

        ## CLIP image-to-image similarity
        # clip_sim_score = clip_image_similarity(original_image, edited_image)
        # clip_sim_score = clip_loss_fn.cnn_feature_loss(original_tensor, edited_tensor).detach().cpu()
        # scores_clip_sim.append(clip_sim_score)

    # Summary
    print("\n=== Averages ===")
    if scores_clip_loss:
        print(f"Mean S_dir (clip_loss):   {np.mean(scores_clip_loss):.4f}")
    # if scores_clip_sim:
    #     print(f"Mean CLIP similarity:     {np.mean(scores_clip_sim):.4f}")

## ====================================
if __name__ == "__main__":

    experiments = [
        {
            "name" : "authors",
            "original_dir" : "/home/ubuntu/controlbfr/asyrp-extension/src/lib/asyrp/runs/smiling_LC_CelebA_HQ_t999_ninv50_ngen50/test_images/50/original",
            "edited_dir" : "/home/ubuntu/controlbfr/asyrp-extension/src/lib/asyrp/runs/smiling_LC_CelebA_HQ_t999_ninv50_ngen50/test_images/50/edited",
            "reconstructed_dir" : "/home/ubuntu/controlbfr/asyrp-extension/src/lib/asyrp/runs/smiling_LC_CelebA_HQ_t999_ninv50_ngen50/test_images/50/reconstructed",
        },
        # {
        #     "name": "exp12_smile",
        #     "original_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp12_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "edited_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp12_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        #     "reconstructed_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp12_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/reconstructed"
        # },
        # {
        #     "name": "exp13_4_smile",
        #     "original_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images_ckpt04/40/original",
        #     "edited_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images_ckpt04/40/edited",
        #     "reconstructed_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images_ckpt04/40/reconstructed"
        # },
        # {
        #     "name": "exp13_3_smile",
        #     "original_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images_ckpt03/40/original",
        #     "edited_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images_ckpt03/40/edited",
        #     "reconstructed_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images_ckpt03/40/reconstructed"
        # },
        # {
        #     "name" : "exp13b_500_smile",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_500_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "edited_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_500_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        #     "reconstructed_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_500_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/reconstructed",
        # },
        # {
        #     "name" : "exp13b_1000_smile",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_1000_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "edited_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_1000_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        # },
        # {
        #     "name"         : "exp13_true_smiling_5",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/5_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/5_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        # },
        # {
        #     "name"         : "exp13_true_smiling_6",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/6_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/6_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        # },
        # {
        #     "name"         : "exp13_true_smiling_7",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/7_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/7_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        # },
        # {
        #     "name"         : "exp13_true_smiling_8",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/8_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/8_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        # },
        # {
        #     "name"         : "exp13_true_smiling_11",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/11_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/11_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        # },
        # {
        #     "name"         : "exp13_true_smiling_12",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/12_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/12_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        # },
        # {
        #     "name"         : "exp13_true_smiling_13",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/13_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/13_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        # },
        # {
        #     "name"         : "exp13_true_smiling_14",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/14_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_smiling/14_smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        # },
        # {
        #     "name" : "exp13_true_fastLR_smiling_100",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_fastLR_2_smiling/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/50/original",
        #     "edited_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_true_fastLR_2_smiling/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/50/edited",
        # },
        {
            "name" : "exp15_smiling",
            "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_smiling/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
            "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_smiling/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        },
        {
            "name" : "exp15_mamba_lr0.45_l12.5_smiling",
            "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_mamba_lr0.45_l12.5_smiling/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
            "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_mamba_lr0.45_l12.5_smiling/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        },
        {
            "name" : "exp15_mamba_0.5_l12.5_smiling",
            "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_mamba_lr0.5_l12.5_smiling/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
            "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_mamba_lr0.5_l12.5_smiling/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        },
        {
            "name" : "exp15_mamba_angry",
            "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_mamba_angry/angry_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
            "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_mamba_angry/angry_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        },
        {
            "name" : "exp15_mamba_sad",
            "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_mamba_sad/sad_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
            "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_mamba_sad/sad_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        },
        {
            "name" : "exp15_mamba_tanned",
            "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_mamba_tanned/tanned_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
            "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_mamba_tanned/tanned_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        },
        {
            "name" : "exp15_mamba_makeup",
            "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_mamba_makeup/makeup_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
            "edited_dir"   : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp15_mamba_makeup/makeup_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        },
        # {
        #     "name" : "exp14_sad",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp14_sad/sad_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "edited_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp14_sad/sad_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        #     "reconstructed_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp14_sad/sad_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/reconstructed",
        # },
    ]

    src_txt = "face"
    trg_txt1 = "smiling face"
    trg_txt2 = "sad face"
    trg_txt3 = "angry face"
    trg_txt4 = "tanned face"
    trg_txt5 = "disgusted face"

    # Process each experiment
    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Processing experiment: {exp['name']}")
        print(f"{'='*50}")

        # Process edited images
        if "smiling" or "smile" in exp['name']:
            main(exp['original_dir'], exp['edited_dir'], src_txt, trg_txt1)
        elif "sad" in exp['name']:
            main(exp['original_dir'], exp['edited_dir'], src_txt, trg_txt2)
        elif "angry" in exp['name']:
            main(exp['original_dir'], exp['edited_dir'], src_txt, trg_txt3)
        elif "tanned" in exp['name']:
            main(exp['original_dir'], exp['edited_dir'], src_txt, trg_txt4)
        elif "disgusted" in exp['name']:
            main(exp['original_dir'], exp['edited_dir'], src_txt, trg_txt5)

# '''
# For authors outputs:
# === Averages ===
# Mean S_dir (ours):        0.0845
# Mean S_dir (clip_loss):   0.1155
# Mean CLIP similarity:     0.8223


# For our exp12:
# === Averages ===
# Mean S_dir (ours):        0.0787
# Mean S_dir (clip_loss):   0.1023
# Mean CLIP similarity:     0.7604

# For our exp13:

# '''







# # main.py
# import torch
# from glob import glob
# from PIL import Image
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parents[1]))
# from losses.clip_loss import CLIPLoss

# def main(original_dir, edited_dir, caption1, caption2):

#    # Initialize using original repo's implementation
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     loss_fn = CLIPLoss(device)

#     original_dir = Path(original_dir)
#     edited_dir = Path(edited_dir)
#     image_paths = sorted(original_dir.glob("*_original.png"))

#     total_sim = 0.0
#     for original_path in image_paths:
#         edited_path = edited_dir / original_path.name.replace("_original", "_edited")
#         if not edited_path.exists():
#             print(f"Missing: {edited_path.name}\n{edited_path}\n")
#             continue

#         orig_img = Image.open(original_path).convert("RGB")
#         edit_img = Image.open(edited_path).convert("RGB")

#         # Use original repo's directional similarity calculation
#         sim = loss_fn.clip_directional_loss(
#             orig_img, 
#             edit_img,
#             [caption1],
#             [caption2]
#         )
#         total_sim += sim.item()
#         c += 1
    
#     avg_sim = total_sim / c
#     print(f"Average Directional CLIP Similarity: {avg_sim:.4f}")
    
#     # # Get sorted image pairs
#     # original_paths = Path(glob(f"{original_dir}/*.png"))
#     # edited_paths = Path(glob(f"{edited_dir}/*.png"))

#     # total_sim = 0.0
#     # for orig_path, edit_path in zip(original_paths, edited_paths):
#     #     # Load images
#     #     orig_img = Image.open(orig_path).convert("RGB")
#     #     edit_img = Image.open(edit_path).convert("RGB")

#     #     # Use original repo's directional similarity calculation
#     #     sim = loss_fn.directional_clip_similarity(
#     #         orig_img, 
#     #         edit_img,
#     #         [caption1],
#     #         [caption2]
#     #     )
#     #     total_sim += sim.item()
    
#     # avg_sim = total_sim / len(original_paths)
#     # print(f"Average Directional CLIP Similarity: {avg_sim:.4f}")

# if __name__ == "__main__":

#     original_dir = "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp14_sad/sad_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original"
#     edited_dir   = "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp14_sad/sad_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited"
#     caption1     = "face"
#     caption2     = "sad face" 
    
#     main(original_dir, edited_dir, caption1, caption2)