import subprocess
import shutil
import time

# ฟังก์ชันช่วยเช็ค-ติดตั้ง ollama CLI และดึงโมเดล gemma3:latest พร้อมสตาร์ทเซิร์ฟเวอร์ถ้ายังไม่รัน
def ensure_ollama_gemma():
    # 1) ติดตั้ง ollama CLI ถ้าไม่มี
    if shutil.which("ollama") is None:
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    # 2) ตรวจสอบว่าเซิร์ฟเวอร์รันอยู่ (ollama list)
    proc = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if proc.returncode != 0 or "could not connect" in proc.stderr.lower():
        # สตาร์ท Ollama server ใน background
        subprocess.Popen([
            "ollama", "serve"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # รอสักครู่ให้เซิร์ฟเวอร์พร้อมรับคำสั่ง
        time.sleep(3)
    # 3) ดึงโมเดล gemma3:latest ถ้ายังไม่มี
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
