import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class StoryboardNode:
    # กำหนดหมวดหมู่เป็น class attribute แทน method เพื่อให้ JSON serializable
    CATEGORY = "Storyboard"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "label": ("STRING", {"default": "Scene 1", "multiline": False}),
                "action": ("STRING", {"default": "", "multiline": True}),
                "camera": ("STRING", {"default": "", "multiline": True}),
                "notes": ("STRING", {"default": "", "multiline": True}),
                "mood": ("STRING", {"default": "", "multiline": True}),
                "dialogue": ("STRING", {"default": "", "multiline": True}),
                "details": ("STRING", {"default": "", "multiline": True}),
                # เพิ่ม input สำหรับข้อความเสริมจาก PromptExtraNode
                "extra_text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"
    OUTPUT_NODE = True

    def __init__(self):
        self.processor = None
        self.model = None

    def _load_model(self):
        if self.processor is None or self.model is None:
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.model = self.model.to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    def generate_caption(
        self,
        image,
        label,
        action,
        camera,
        notes,
        mood,
        dialogue,
        details,
        extra_text,
    ):
        # โหลดโมเดลและสร้าง caption
        self._load_model()
        # แปลง Tensor → NumPy → PIL Image
        np_img = (image.cpu().numpy() * 255).astype("uint8")
        pil_image = Image.fromarray(np_img).convert("RGB")

        inputs = self.processor(pil_image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        # รวม prompt หลักกับข้อความเสริม
        prompt_parts = []
        if extra_text:
            prompt_parts.append(f"Extra Text: {extra_text}")
        prompt_parts.extend([
            f"Label: {label}",
            f"Caption: {caption}",
            f"Action: {action}",
            f"Camera: {camera}",
            f"Notes: {notes}",
            f"Mood: {mood}",
            f"Dialogue: {dialogue}",
            f"Details: {details}",
        ])
        full_prompt = "\n".join(prompt_parts)
        return (full_prompt,)

# ข้อความเพื่อยืนยันการโหลดโมดูล
print("📦 storyboard_node module loaded")

# กำหนด mapping ให้ ComfyUI ใช้งาน
NODE_CLASS_MAPPINGS = {
    "StoryboardNode": StoryboardNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StoryboardNode": "🎬 Storyboard Image → Prompt"
}
print("✅ NODE_CLASS_MAPPINGS defined")
