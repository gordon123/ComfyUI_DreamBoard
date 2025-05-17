import torch
from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration

CATEGORY = "Storyboard"

class StoryboardNode:
    def __init__(self):
        self.processor = None
        self.model = None

    def _load_model(self):
        if self.processor is None or self.model is None:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def CATEGORY(cls):
        return CATEGORY

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

    def generate_caption(self, image, label, action, camera, notes, mood, dialogue, details):
        self._load_model()
        pil_image = Image.fromarray((image * 255).astype("uint8")).convert("RGB")
        inputs = self.processor(pil_image, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        prompt = f"Label: {label}\nCaption: {caption}\nAction: {action}\nCamera: {camera}\nNotes: {notes}\nMood: {mood}\nDialogue: {dialogue}\nDetails: {details}"
        return (prompt,)

print("📦 storyboard_node module loaded")

NODE_CLASS_MAPPINGS = {
    "StoryboardNode": StoryboardNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StoryboardNode": "🎬 Storyboard Image → Prompt"
}
print("✅ NODE_CLASS_MAPPINGS defined")
