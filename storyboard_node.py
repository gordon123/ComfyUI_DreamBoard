
import torch
from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration

CATEGORY = "Storyboard"

class StoryboardNode:
    def __init__(self):
        # Load the image captioning model (BLIP in this case)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input from another node
                "label": ("STRING", {"multiline": False}),
                "action": ("STRING", {"multiline": True}),
                "camera": ("STRING", {"multiline": True}),
                "notes": ("STRING", {"multiline": True}),
                "mood": ("STRING", {"multiline": True}),
                "dialogue": ("STRING", {"multiline": True}),
                "details": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"
    OUTPUT_NODE = True

    def generate_caption(self, image, label, action, camera, notes, mood, dialogue, details):
        pil_image = Image.fromarray((image * 255).astype("uint8")).convert("RGB")

        # Use BLIP to generate caption
        inputs = self.processor(pil_image, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        # You could combine all fields here into a rich prompt if needed
        prompt = f"Label: {label}\nCaption: {caption}\nAction: {action}\nCamera: {camera}\nNotes: {notes}\nMood: {mood}\nDialogue: {dialogue}\nDetails: {details}"
        return (prompt,)

NODE_CLASS_MAPPINGS = {
    "StoryboardNode": StoryboardNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StoryboardNode": "🎬 Storyboard Image → Prompt"
}
