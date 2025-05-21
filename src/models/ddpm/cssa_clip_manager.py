# cssa_clip_manager.py
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

class SemanticMemoryManager:
    def __init__(self, device='cuda', model_name="openai/clip-vit-base-patch16"):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.memory_bank = []

    def encode(self, image_tensor):
        """
        Convert a torch image tensor [B, 3, H, W] to CLIP semantic embeddings.
        """
        image_tensor = image_tensor.clamp(0, 1)
        images = [transforms.ToPILImage()(img.cpu()) for img in image_tensor]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            embeddings = self.clip_model.get_image_features(**inputs)
        return embeddings  # [B, 512]

    def update_memory(self, embedding):
        self.memory_bank.append(embedding)
        if len(self.memory_bank) > 5:  # optional window size
            self.memory_bank.pop(0)

    def get_memory(self):
        return self.memory_bank