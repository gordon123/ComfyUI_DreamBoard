from ollama import Ollama

class PromptExtraNode:
    """
    Node สำหรับรับ text แล้วส่งให้โมเดล Ollama gemma3:latest ประมวลผล
    """
    CATEGORY = "Storyboard"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "pass_text"
    OUTPUT_NODE = False

    def __init__(self):
        # สร้าง Ollama client เพื่อเรียก API
        self.client = Ollama()

    def pass_text(self, text):
        # เรียก Ollama Python client รันโมเดล
        response = self.client.run(
            model="gemma3:latest",
            prompt=text,
            stream=False,
        )
        # คืนข้อความที่โมเดลตอบกลับ
        return (response.text,)

print("📦 prompt_extra_node module loaded")
NODE_CLASS_MAPPINGS = {"PromptExtraNode": PromptExtraNode}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptExtraNode": "Prompt extra"}
print("✅ NODE_CLASS_MAPPINGS defined")
