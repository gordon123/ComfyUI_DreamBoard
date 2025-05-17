import subprocess, shutil

# ฟังก์ชันช่วยเช็ค-ติดตั้ง ollama CLI และดึงโมเดล gemma3:latest
def ensure_ollama_gemma():
    # ติดตั้ง ollama CLI ถ้าไม่มี (Linux install script)
    if shutil.which("ollama") is None:
        subprocess.run("curl -fsSL https://ollama.com/install/linux | sh", shell=True, check=True)
    # ดึงรายชื่อโมเดลที่มีอยู่
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    # ดึงโมเดล gemma3:latest ถ้ายังไม่อยู่
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
        # ตรวจสอบและติดตั้ง ollama+gemma3 ครั้งแรก
        ensure_ollama_gemma()

    def pass_text(self, text):
        # รัน gemma3:latest ผ่าน CLI และรับผลลัพธ์เป็นข้อความ
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
