import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class StoryboardNode:
    """
    Node สำหรับสร้าง prompt จากภาพ พร้อมข้อความเสริม
    """
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
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")

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
        # โหลดโมเดลถ้ายังไม่โหลด
        self._load_model()

        # แปลง image เป็น PIL Image รองรับหลากหลาย shape
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            # เป็น Tensor หรือ array
            if isinstance(image, torch.Tensor):
                arr = image.cpu().numpy()
            elif isinstance(image, np.ndarray):
                arr = image
            else:
                arr = np.array(image)

            # ลดมิติจาก batch/frame ถ้ามี
            while arr.ndim > 3 and arr.shape[0] == 1:
                arr = arr[0]

            # ตรวจสอบ HWC หรือ CHW
            if arr.ndim == 3 and arr.shape[2] == 3:
                pass
            elif arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            else:
                raise TypeError(f"Unsupported image shape: {arr.shape}")

            # แปลงค่าเป็น 0-255 uint8
            arr = (arr * 255).clip(0, 255).astype("uint8")
            pil_image = Image.fromarray(arr).convert("RGB")

        # สร้าง caption จาก BLIP
        inputs = self.processor(pil_image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        # สร้าง prompt เต็ม
        parts = []
        if extra_text:
            parts.append(f"Extra Text: {extra_text}")
        parts.extend([
            f"Label: {label}",
            f"Caption: {caption}",
            f"Action: {action}",
            f"Camera: {camera}",
            f"Notes: {notes}",
            f"Mood: {mood}",
            f"Dialogue: {dialogue}",
            f"Details: {details}",
        ])
        full_prompt = "\n".join(parts)
        return (full_prompt,)

# ยืนยันการโหลดโมดูล
print("📦 storyboard_node module loaded")
NODE_CLASS_MAPPINGS = {"StoryboardNode": StoryboardNode}
NODE_DISPLAY_NAME_MAPPINGS = {"StoryboardNode": "🎬 Storyboard Image → Prompt"}
print("✅ NODE_CLASS_MAPPINGS defined")
