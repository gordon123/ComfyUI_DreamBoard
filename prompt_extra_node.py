class PromptExtraNode:
    """
    A simple node that lets you enter arbitrary text and outputs it as a STRING.
    Appears as 'Prompt extra' in the node list.
    """
    CATEGORY = "Storyboard"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # A multiline text field for extra prompt text
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "pass_text"
    OUTPUT_NODE = False

    def pass_text(self, text):
        # Simply return the entered text
        return (text,)

# Notify ComfyUI this module is loaded
print("📦 prompt_extra_node module loaded")

# Map the class to its node name
NODE_CLASS_MAPPINGS = {
    "PromptExtraNode": PromptExtraNode
}

# Set the display name shown in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptExtraNode": "Prompt extra"
}
print("✅ NODE_CLASS_MAPPINGS defined")
