import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt", antialias=True)["pixel_values"] #, do_resize=True, size=224, resample=3, do_center_crop=True, crop_size=224, do_normalize=True, do_rescale=True, rescale_factor=1/255, do_pad=True, do_convert_rgb=True, antialias=True)["pixel_values"]
        return {"pixel_values": image.to("cuda")}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to("cuda")}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def image_similarity(self, image_one, image_two):
        """
        Compute CLIP cosine similarity between two images.
        This measures how semantically similar the two images are.
        """
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)

        sim_score = F.cosine_similarity(img_feat_one, img_feat_two).item()
        return sim_score

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity

##############
## Version 1
##############

def main(original_dir, edited_dir, caption1, caption2):
    original_dir = Path(original_dir)
    edited_dir = Path(edited_dir)

    clip_id = "openai/clip-vit-base-patch32"  # Or use large-patch14 if needed
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to("cuda")
    image_processor = CLIPImageProcessor.from_pretrained(clip_id)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to("cuda")

    # Initialize model
    dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)

    scores = []
    image_paths = sorted(original_dir.glob("*_original.png"))
    edited_paths = sorted(edited_dir.glob("*_edited.png")) 

    for original_path, edited_path in zip(image_paths, edited_paths):
        # print(f"Original Path: {original_path}\nEdited Path: {edited_path}\n")
        # edited_path = edited_dir / original_path.name.replace("_original", "_edited")
        # edited_path = edited_dir / original_path.name.replace("_original", "_reconstructed")

        if not edited_path.exists():
            print(f"Missing: {edited_path.name}\n{edited_path}\n")
            continue

        original_image = Image.open(original_path).convert("RGB")
        edited_image = Image.open(edited_path).convert("RGB")

        similarity_score = dir_similarity(original_image, edited_image, caption1, caption2)

        # Convert to scalar and store
        scores.append(float(similarity_score.detach().cpu()))
        # print(f"{original_path.name} → S_dir: {similarity_score.item():.4f}")
        # clip_sim_score = dir_similarity.image_similarity(original_image, edited_image)
        # print(f"{original_path.name} → CLIP similarity: {clip_sim_score:.4f}")

    if scores:
        print(f"\nMean S_dir over {len(scores)} pairs: {np.mean(scores):.4f}")
    else:
        print("No valid image pairs found.")

    # if clip_sim_score:
    #     print(f"\nMean CLIP similarity over {len(scores)} pairs: {np.mean(clip_sim_score):.4f}")
    # else:
    #     print("No valid image pairs found.")

##############
## Version 2
##############

# sys.path.append(str(Path(__file__).resolve().parents[1]))
# from losses.clip_loss import CLIPLoss
# import torchvision.transforms as transforms

# def main(original_dir, edited_dir, caption1, caption2):
#     original_dir = Path(original_dir)
#     edited_dir = Path(edited_dir)

#     clip_id = "openai/clip-vit-base-patch32"
#     tokenizer = CLIPTokenizer.from_pretrained(clip_id)
#     text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to("cuda")
#     image_processor = CLIPImageProcessor.from_pretrained(clip_id)
#     image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to("cuda")

#     ## DirectionalSimilarity setup
#     dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)

#     ## CLIPLoss setup
#     clip_loss_fn = CLIPLoss(device="cuda", lambda_direction=1.0)
#     clip_loss_fn.set_text_features(caption1, caption2)
#     transform = clip_loss_fn.clip_preprocess  # Use the same preprocessing as the model expects

#     ## Score lists
#     scores_dir_sim = []
#     scores_clip_sim = []
#     scores_clip_loss = []

#     image_paths = sorted(original_dir.glob("*_original.png"))

#     for original_path in image_paths:
#         edited_path = edited_dir / original_path.name.replace("_original", "_edited")
#         if not edited_path.exists():
#             print(f"Missing: {edited_path.name}\n{edited_path}\n")
#             continue

#         ## Load images
#         original_image = Image.open(original_path).convert("RGB")
#         edited_image = Image.open(edited_path).convert("RGB")

#         ## Directional Sdir (image-text)
#         # sdir_score = dir_similarity(original_image, edited_image, caption1, caption2)
#         # scores_dir_sim.append(float(sdir_score.detach().cpu()))

#         ## Author's CLIP directional loss-based Sdir
#         original_tensor = transform(original_image).unsqueeze(0).to("cuda")
#         edited_tensor = transform(edited_image).unsqueeze(0).to("cuda")
#         # sdir_clip_loss = 1.0 - clip_loss_fn.clip_directional_loss(original_tensor, caption1, edited_tensor, caption2).item()
#         sdir_clip_loss = clip_loss_fn.clip_directional_loss(original_tensor, caption1, edited_tensor, caption2).item()
#         scores_clip_loss.append(sdir_clip_loss)

#         ## CLIP image-to-image similarity
#         # clip_sim_score = dir_similarity.image_similarity(original_image, edited_image)
#         # scores_clip_sim.append(clip_sim_score)

#         # Print per image
#         # print(f"{original_path.name} → S_dir (ours): {sdir_score.item():.4f}, "
#         #       f"CLIP similarity: {clip_sim_score:.4f}, "
#         #       f"S_dir (clip_loss): {sdir_clip_loss:.4f}")

#     # Summary
#     print("\n=== Averages ===")
#     # if scores_dir_sim:
#     #     print(f"Mean S_dir (ours):        {np.mean(scores_dir_sim):.4f}")
#     if scores_clip_loss:
#         print(f"Mean S_dir (clip_loss):   {np.mean(scores_clip_loss):.4f}")
#     if scores_clip_sim:
#         print(f"Mean CLIP similarity:     {np.mean(scores_clip_sim):.4f}")

## ====================================
if __name__ == "__main__":

    experiments = [
        # {
        #     "name" : "authors",
        #     "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/40/original",
        #     "edited_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/40/edited",
        #     "reconstructed_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/40/reconstructed"
        # },
        # {
        #     "name" : "authors",
        #     "original_dir" : "/home/ubuntu/controlbfr/asyrp-extension/src/lib/asyrp/runs/smiling_LC_CelebA_HQ_t999_ninv50_ngen50/test_images/50/original",
        #     "edited_dir" : "/home/ubuntu/controlbfr/asyrp-extension/src/lib/asyrp/runs/smiling_LC_CelebA_HQ_t999_ninv50_ngen50/test_images/50/edited",
        #     "reconstructed_dir" : "/home/ubuntu/controlbfr/asyrp-extension/src/lib/asyrp/runs/smiling_LC_CelebA_HQ_t999_ninv50_ngen50/test_images/50/reconstructed"
        # },
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
        #     # "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_1000_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
        #     "original_dir" : "/home/ubuntu/controlbfr/asyrp-extension/src/lib/asyrp/runs/smiling_LC_CelebA_HQ_t999_ninv50_ngen50/test_images/50/original",
        #     "edited_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_1000_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        #     "reconstructed_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_1000_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/reconstructed",
        # },
        {
            "name" : "exp14_sad",
            "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp14_sad/sad_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
            "edited_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp14_sad/sad_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
            "reconstructed_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp14_sad/sad_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/reconstructed",
        }
    ]

    caption1 = "face"
    caption2 = "smiling face"
    caption3 = "sad face"

    # Process each experiment
    for exp in experiments:
        print(f"\n{'='*35}")
        print(f"Processing experiment: {exp['name']}")
        print(f"{'='*35}")

        # Process edited images
        # print("\nEvaluating edited images:")
        # main(exp['original_dir'], exp['edited_dir'], caption1, caption2)
        main(exp['original_dir'], exp['edited_dir'], caption1, caption3)

        # Process reconstructed images if needed
        # print("\nEvaluating reconstructed images:")
        # main(exp['original_dir'], exp['reconstructed_dir'], caption1, caption2)


'''
For authors outputs:
=== Averages ===
Mean S_dir (ours):        0.0845
Mean CLIP similarity:     0.8223
Mean S_dir (clip_loss):   0.1155

For our exp12:
=== Averages ===
Mean S_dir (ours):        0.0787
Mean CLIP similarity:     0.7604
Mean S_dir (clip_loss):   0.1023

For our exp13:

'''