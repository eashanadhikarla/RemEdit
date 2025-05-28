# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

# from act.utils import utils

from pathlib import Path
from PIL import Image
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)-12s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

import torch.nn as nn
import torch.nn.functional as F

# https://huggingface.co/docs/diffusers/en/conceptual/evaluation
class DirectionalSimilarity(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        clip_id = "openai/clip-vit-large-patch14"
        # clip_id = "openai/clip-vit-large-patch16"

        # clip_id = "openai/clip-vit-base-patch16"
        # clip_id = "openai/clip-vit-base-patch32"

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_id)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(
            device
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(
            device
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_id)

        self.device = device

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(self.device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs.input_ids.to(self.device),
            "attention_mask": inputs.attention_mask.to(self.device),
        }

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

    def compute_directional_similarity(
        self, img_feat_one, img_feat_two, text_feat_one, text_feat_two
    ):
        sim_direction = F.cosine_similarity(
            img_feat_two - img_feat_one, text_feat_two - text_feat_one
        )
        return sim_direction

    @torch.inference_mode()
    def forward(
        self,
        image_one,
        image_two,
        caption_one,
        caption_two,
        caption_zero_shot_one,
        caption_zero_shot_two,
    ):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        text_feat_zero_shot_one = self.encode_text(caption_zero_shot_one)
        text_feat_zero_shot_two = self.encode_text(caption_zero_shot_two)
        text_similarity = (
            F.cosine_similarity(text_feat_one, text_feat_two).detach().cpu().numpy()
        )
        image_similarity = (
            F.cosine_similarity(img_feat_one, img_feat_two).detach().cpu().numpy()
        )
        conditional_similarity = (
            F.cosine_similarity(img_feat_two, text_feat_two).detach().cpu().numpy()
        )
        unconditional_similarity = (
            F.cosine_similarity(img_feat_two, text_feat_one).detach().cpu().numpy()
        )
        directional_similarity = (
            self.compute_directional_similarity(
                img_feat_one, img_feat_two, text_feat_one, text_feat_two
            )
            .detach()
            .cpu()
            .numpy()
        )
        unconditional_zero_shot_similarity = F.cosine_similarity(
            img_feat_two, text_feat_zero_shot_one
        )
        conditional_zero_shot_similarity = F.cosine_similarity(
            img_feat_two, text_feat_zero_shot_two
        )
        zero_shot_score = (
            F.softmax(
                torch.stack(
                    [
                        unconditional_zero_shot_similarity,
                        conditional_zero_shot_similarity,
                    ],
                    dim=1,
                ),
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()[:, 1]
        )
        return {
            "text_similarity": text_similarity,
            "image_similarity": image_similarity,
            "conditional_similarity": conditional_similarity,
            "unconditional_similarity": unconditional_similarity,
            "directional_similarity": directional_similarity,
            "conditional_zero_shot_score": zero_shot_score,
        }


def calculate_clip_score(cfg: DictConfig) -> None:
    """
    Main function to calculate CLIP scores for images based on prompts from JSON files or command line arguments.

    This function handles the parsing of command line arguments, reading of image data, and calculation of CLIP scores using zero-shot learning.

    Args:
        args (argparse.Namespace): The parsed command line arguments containing input folder path and prompt field.
    """
    meta_dict = defaultdict(list)
    logger.info(f"Processing directory: {cfg.input_folder}")
    for img_path in sorted(Path(cfg.input_folder).glob("**/*.png")):
        # images += [Image.open(img_path)]
        with (Path(img_path).with_suffix(".json")).open("r") as fp:
            meta = json.load(fp)
            for k, v in meta.items():
                if isinstance(v, list):
                    meta_dict[k].extend(v)
                else:
                    meta_dict[k].append(v)
    df = pd.DataFrame(meta_dict)
    assert len(df) > 0, "No images found in input folder."
    similarity = DirectionalSimilarity(cfg.device)
    results = []
    for id in df["id"].unique():
        df_id = df[df["id"] == id]
        unconditional_image_data = df_id[df_id["strength"] == 0]
        unconditional_image = Image.open(unconditional_image_data["image_path"].iloc[0])
        unconditional_prompt = [unconditional_image_data["original_prompt"].iloc[0]]
        conditional_images = []
        conditional_prompts = []
        conditional_zero_shot_prompt = []
        unconditional_zero_shot_prompt = []
        for idx, row in df_id.iterrows():
            condition = (
                row["src_subsets"]
                if "none" in row["dst_subsets"]
                else row["dst_subsets"]
            )
            conditional_images += [Image.open(row["image_path"])]
            conditional_prompts += [row["conditional_prompt"]]
            conditional_zero_shot_prompt += [
                f"A picture of {condition.replace('_', ' ').replace('-', ' ')}."
            ]
            unconditional_zero_shot_prompt += [f"A picture of something."]
        clip_score = similarity.forward(
            unconditional_image,
            conditional_images,
            unconditional_prompt,
            conditional_prompts,
            unconditional_zero_shot_prompt,
            conditional_zero_shot_prompt,
        )
        for k, v in clip_score.items():
            df_id[k] = v
        results += [df_id]
    results = pd.concat(results)
    if cfg.results_dir is not None:
        output_path = Path(Path(__file__).stem)
        output_path = Path(cfg.results_dir, output_path)
        output_path.mkdir(exist_ok=True, parents=True)
    else:
        output_path = None
    if output_path is not None:
        results.to_csv(Path(output_path) / "clip_score.csv")
    return results

def main(original_dir, edited_dir, caption1, caption2):
    original_dir = Path(original_dir)
    edited_dir = Path(edited_dir)

    # Initialize DirectionalSimilarity
    clip_similarity = DirectionalSimilarity(device="cuda")

    scores_directional = []
    scores_img_sim = []
    scores_text_sim = []
    scores_conditional = []
    scores_unconditional = []
    scores_zero_shot = []

    image_paths = sorted(original_dir.glob("*_original.png"))

    for original_path in image_paths:
        edited_path = edited_dir / original_path.name.replace("_original", "_edited")
        if not edited_path.exists():
            print(f"Missing: {edited_path.name}")
            continue

        # Load images
        original_image = Image.open(original_path).convert("RGB")
        edited_image = Image.open(edited_path).convert("RGB")

        # Form zero-shot prompts (can be customized)
        zero_shot_src = ["A picture of something."]
        zero_shot_tgt = [f"A picture of {caption2}."]

        # Compute similarity metrics
        clip_scores = clip_similarity.forward(
            original_image,
            edited_image,
            [caption1],
            [caption2],
            zero_shot_src,
            zero_shot_tgt
        )

        # Store results
        scores_directional.append(clip_scores["directional_similarity"].item())
        scores_img_sim.append(clip_scores["image_similarity"].item())
        scores_text_sim.append(clip_scores["text_similarity"].item())
        scores_conditional.append(clip_scores["conditional_similarity"].item())
        scores_unconditional.append(clip_scores["unconditional_similarity"].item())
        scores_zero_shot.append(clip_scores["conditional_zero_shot_score"].item())

    # (Optional) return results for aggregation/logging
    return {
        "directional_similarity": scores_directional,
        "image_similarity": scores_img_sim,
        "text_similarity": scores_text_sim,
        "conditional_similarity": scores_conditional,
        "unconditional_similarity": scores_unconditional,
        "zero_shot_score": scores_zero_shot,
    }

def run_experiments():
    experiments = [
        {
            "name": "authors",
            "original_dir": "/home/ubuntu/controlbfr/asyrp-extension/src/lib/asyrp/runs/smiling_LC_CelebA_HQ_t999_ninv50_ngen50/test_images/50/original",
            "edited_dir": "/home/ubuntu/controlbfr/asyrp-extension/src/lib/asyrp/runs/smiling_LC_CelebA_HQ_t999_ninv50_ngen50/test_images/50/edited"
        },
        {
            "name": "exp12_smile",
            "original_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp12_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
            "edited_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp12_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        },
        {
            "name": "exp13_4_smile",
            "original_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images_ckpt04/40/original",
            "edited_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images_ckpt04/40/edited",
        },
        {
            "name": "exp13_3_smile",
            "original_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images_ckpt03/40/original",
            "edited_dir": "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images_ckpt03/40/edited",
        },
        {
            "name" : "exp13b_500_smile",
            "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_500_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
            "edited_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_500_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        },
        {
            "name" : "exp13b_1000_smile",
            "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_1000_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
            "edited_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp13b_1000_smile/smiling_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        },
        {
            "name" : "exp14_sad",
            "original_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp14_sad/sad_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/original",
            "edited_dir" : "/home/ubuntu/controlbfr/RiemannianEdit/src/runs/exp14_sad/sad_LC_CelebA_HQ_t999_ninv50_ngen40/test_images/40/edited",
        }
    ]

    # Source and target captions
    src_txt = "face"
    attributes = {
        "smiling": "smiling face",
        "sad": "sad face",
        "angry": "angry face",
        "tanned": "tanned face",
        "disgusted": "disgusted face"
    }

    for exp in experiments:
        print(f"\n{'=' * 50}")
        print(f"🧪 Processing experiment: {exp['name']}")
        print(f"{'=' * 50}")

        # Extract attribute from the original_dir path
        target_caption = None
        for key in attributes:
            if key in exp["original_dir"]:
                target_caption = attributes[key]
                break
        if not target_caption:
            print(f"⚠️  No matching attribute found in path: {exp['original_dir']}")
            continue

        # Run directional similarity evaluation
        results = main(exp["original_dir"], exp["edited_dir"], src_txt, target_caption)

        # Print results
        print(f"\n🔍 Results for: {exp['name']}")
        print(f"Directional Similarity (mean): {torch.tensor(results['directional_similarity']).mean():.4f}")
        print(f"Image Similarity (mean):       {torch.tensor(results['image_similarity']).mean():.4f}")
        print(f"Text Similarity (mean):        {torch.tensor(results['text_similarity']).mean():.4f}")
        print(f"Conditional Sim. (mean):       {torch.tensor(results['conditional_similarity']).mean():.4f}")
        print(f"Unconditional Sim. (mean):     {torch.tensor(results['unconditional_similarity']).mean():.4f}")
        print(f"Zero-shot Score (mean):        {torch.tensor(results['zero_shot_score']).mean():.4f}")

if __name__ == "__main__":
    run_experiments()