import re
import json
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import ollama

class StoryboardNode:
    """
    Node สำหรับสร้าง prompt จากภาพ พร้อมแบ่งท่อนเนื้อเพลงตาม label
    และสร้างฟิลด์ต่าง ๆ โดยใช้ BLIP + Ollama
    """
    CATEGORY = "Storyboard"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "label": ("STRING", {"default": "[Intro solo- Ambient, horror Waterphone Swell]", "multiline": False}),
                "extra_text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    # คืนค่าทั้ง caption, action, camera, notes, mood, dialogue, details
    RETURN_TYPES = ("STRING",) * 7
    RETURN_NAMES = ("caption", "action", "camera", "notes", "mood", "dialogue", "details")
    FUNCTION = "generate_storyboard"
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
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)

    def generate_storyboard(self, image, label, extra_text):
        # 1) สร้าง caption ด้วย BLIP
        self._load_model()
        # ปรับภาพเป็น PIL
        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            arr = image.cpu().numpy() if isinstance(image, torch.Tensor) else np.array(image)
            # ลดมิติ batch/frame
            while arr.ndim > 3 and arr.shape[0] == 1:
                arr = arr[0]
            # CHW -> HWC
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            # แปลงค่าเป็น uint8
            arr = (arr * 255).clip(0, 255).astype("uint8")
            pil = Image.fromarray(arr).convert("RGB")

        inputs = self.processor(pil, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        # 2) ตัด segment ตาม label
        tag = re.escape(label.strip("[]"))
        pattern = rf"\[{tag}\](.*?)(?=\[|\Z)"
        m = re.search(pattern, extra_text, re.S)
        segment = m.group(1).strip() if m else extra_text.strip()

        # 3) สร้าง prompt ให้ Ollama ตอบเฉพาะ JSON
        prompt = (
            f"Image caption: {caption}\n"
            f"Song segment: {segment}\n\n"
            "Please respond ONLY with a JSON object containing only these keys: "
            "action, camera, notes, mood, dialogue, details. "
            "Do not include any explanatory text or code fences."
        )
        print(f"[DEBUG] Ollama prompt: {prompt}")

        # 4) เรียก Ollama
        resp = ollama.generate(model="gemma3:latest", prompt=prompt, stream=False)
        text = resp.get("text", "{}")
        print(f"[DEBUG] Ollama response: {text}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {}

        # 5) คืนค่าทุกช่อง
        return (
            caption,
            data.get("action", ""),
            data.get("camera", ""),
            data.get("notes", ""),
            data.get("mood", ""),
            data.get("dialogue", ""),
            data.get("details", ""),
        )

# ยืนยันการโหลดโมดูล
print("📦 storyboard_node module loaded")
NODE_CLASS_MAPPINGS = {"StoryboardNode": StoryboardNode}
NODE_DISPLAY_NAME_MAPPINGS = {"StoryboardNode": "🎬 Storyboard Image → Prompt"}
print("✅ NODE_CLASS_MAPPINGS defined")
