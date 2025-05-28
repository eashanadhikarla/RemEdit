# https://github.com/athenas-lab/DiffSign/blob/464bc4f1a72f4f9aa10fae87de29ba6dbd549f71/compute_dir_sim.py#L20


import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import glob
from PIL import Image

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor
)

""" 
    Consider the pose transfer from human to synthetic signer as a style transfer problem
    and compute the directional similarity using CLIP to encode text and images.
"""

class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(device)}

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

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity

def computeDirSim():
    """ 
    The pose transfer from human to synthetic signer is treated as an image editing problem.
    Directional similarity measures how well the change in text prompt aligns with the
    change in the generated video with respect to the original human signer video. 
    """

    dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)
    scores = []

    ## Path
    # human_dir = "/home/ubuntu/controlbfr/asyrp-extension/src/lib/asyrp/runs/smiling_LC_CelebA_HQ_t999_ninv50_ngen50/test_images/50/original"
    # synth_dir = "/home/ubuntu/controlbfr/asyrp-extension/src/lib/asyrp/runs/smiling_LC_CelebA_HQ_t999_ninv50_ngen50/test_images/50/edited"

    human_dir = "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_1000_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original"
    synth_dir = "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_1000_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited"

    real_list = sorted(glob.glob(human_dir + "/*.png"))
    fake_list = sorted(glob.glob(synth_dir + "/*.png"))
    print(f"Total images (original/edited): {len(real_list), len(fake_list)}")

    for i in range(len(real_list)):
        original_image = Image.open(real_list[i])
        original_caption = "face"
        edited_image = Image.open(fake_list[i])
        modified_caption = "smiling face"

        ## Compute the directional similarity score frame-by-frame for aaveraging
        similarity_score = dir_similarity(original_image, edited_image, original_caption, modified_caption)
        scores.append(float(similarity_score.detach().cpu()))

    ## Output the mean score
    print(f"CLIP directional similarity: {np.mean(scores):4}") 

    return

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_id = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
    image_processor = CLIPImageProcessor.from_pretrained(clip_id)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)     

    computeDirSim()