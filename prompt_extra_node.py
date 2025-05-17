import subprocess
import shutil

# ฟังก์ชันช่วยเช็ค-ติดตั้ง ollama CLI และดึงโมเดล gemma3:latest
def ensure_ollama_gemma():
    # ติดตั้ง ollama CLI ถ้าไม่มี
    if shutil.which("ollama") is None:
        # ใช้สคริปต์ติดตั้งอย่างเป็นทางการ
        subprocess.run("curl -fsSL https://ollama.com/install/linux | sh", shell=True, check=True)
    # ตรวจสอบรายชื่อโมเดลที่มีอยู่
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    # ดึงโมเดล gemma3:latest ถ้ายังไม่มี
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
        # ตรวจสอบและติดตั้ง ollama+โมเดลครั้งแรก
        ensure_ollama_gemma()

    def pass_text(self, text):
        # เรียกใช้ ollama CLI เพื่อรันโมเดล gemma3:latest
        proc = subprocess.run(
            ["ollama", "run", "gemma3:latest", "--no-stream", "--prompt", text],
            capture_output=True, text=True
        )
        output = proc.stdout.strip()
        return (output,)

# แจ้ง ComfyUI ว่ามอดูลโหลดแล้ว
print("📦 prompt_extra_node module loaded")
NODE_CLASS_MAPPINGS = {"PromptExtraNode": PromptExtraNode}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptExtraNode": "Prompt extra"}
print("✅ NODE_CLASS_MAPPINGS defined")
