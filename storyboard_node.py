import torch
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
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_caption(self, image, label, action, camera, notes, mood, dialogue, details, extra_text):
        # โหลดโมเดลและสร้าง caption
        self._load_model()
        pil_image = Image.fromarray((image * 255).astype("uint8")).convert("RGB")
        inputs = self.processor(pil_image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        # รวม prompt หลักกับข้อความเสริม
        full_prompt = ''
        if extra_text:
            full_prompt += f"Extra Text: {extra_text}\n"
        full_prompt += (
            f"Label: {label}\n"
            f"Caption: {caption}\n"
            f"Action: {action}\n"
            f"Camera: {camera}\n"
            f"Notes: {notes}\n"
            f"Mood: {mood}\n"
            f"Dialogue: {dialogue}\n"
            f"Details: {details}"
        )
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
