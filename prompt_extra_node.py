# prompt_extra_node.py
import subprocess, shutil

def ensure_ollama_gemma():
    # ติดตั้ง ollama CLI ถ้าไม่มี
    if shutil.which("ollama") is None:
        subprocess.run(["pip", "install", "ollama"], check=True)
    # ดึงโมเดลถ้ายังไม่มี
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if "gemma3:latest" not in result.stdout:
        subprocess.run(["ollama", "pull", "gemma3:latest"], check=True)

class PromptExtraNode:
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
        ensure_ollama_gemma()

    def pass_text(self, text):
        # รัน gemma3:latest ผ่าน CLI รับผลลัพธ์เป็นข้อความ
        proc = subprocess.run(
            ["ollama", "run", "gemma3:latest", "--no-stream", "--prompt", text],
            capture_output=True, text=True
        )
        output = proc.stdout.strip()
        return (output,)

print("📦 prompt_extra_node module loaded")
NODE_CLASS_MAPPINGS = {"PromptExtraNode": PromptExtraNode}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptExtraNode": "Prompt extra"}
print("✅ NODE_CLASS_MAPPINGS defined")
