# storyboard_node.py
import subprocess, shutil
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ฟังก์ชันช่วยเช็ค-ติดตั้ง ollama CLI และดึงโมเดล gemma3:latest
def ensure_ollama_gemma():
    # ติดตั้ง ollama CLI ถ้าไม่มี
    if shutil.which("ollama") is None:
        subprocess.run(["pip", "install", "ollama"], check=True)
    # ดึงรายชื่อโมเดลที่มีอยู่
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if "gemma3:latest" not in result.stdout:
        subprocess.run(["ollama", "pull", "gemma3:latest"], check=True)

class StoryboardNode:
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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"
    OUTPUT_NODE = True

    def __init__(self):
        # เตรียม Ollama+GEMMA3 ก่อนครั้งแรก
        ensure_ollama_gemma()
        self.processor = None
        self.model = None

    def _load_model(self):
        if self.processor is None or self.model is None:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_caption(self, image, label, action, camera, notes, mood, dialogue, details):
        self._load_model()
        pil_image = Image.fromarray((image * 255).astype("uint8")).convert("RGB")
        inputs = self.processor(pil_image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        prompt = (
            f"Label: {label}\n"
            f"Caption: {caption}\n"
            f"Action: {action}\n"
            f"Camera: {camera}\n"
            f"Notes: {notes}\n"
            f"Mood: {mood}\n"
            f"Dialogue: {dialogue}\n"
            f"Details: {details}"
        )
        return (prompt,)

print("📦 storyboard_node module loaded")
NODE_CLASS_MAPPINGS = {"StoryboardNode": StoryboardNode}
NODE_DISPLAY_NAME_MAPPINGS = {"StoryboardNode": "🎬 Storyboard Image → Prompt"}
print("✅ NODE_CLASS_MAPPINGS defined")
