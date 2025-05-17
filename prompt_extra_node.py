import os
import subprocess

# กำหนดตำแหน่งโฟลเดอร์และ binary ของ Ollama ภายในโฟลเดอร์นี้
BASE_DIR = os.path.dirname(__file__)
OLLAMA_DIR = os.path.join(BASE_DIR, "ollama")
OLLAMA_BIN = os.path.join(OLLAMA_DIR, "ollama")

# ฟังก์ชันช่วยเช็ค-ติดตั้ง ollama CLI และดึงโมเดล gemma3:latest

def ensure_ollama_gemma():
    # สร้างโฟลเดอร์ ollama ถ้ายังไม่มี
    os.makedirs(OLLAMA_DIR, exist_ok=True)
    # ถ้าไม่มีไฟล์ binary ให้ดาวน์โหลดและแตกไฟล์
    if not os.path.isfile(OLLAMA_BIN):
        url = (
            "https://github.com/jmorganca/ollama/releases/latest/"
            "download/ollama-linux-amd64.tar.gz"
        )
        # ดาวน์โหลดและแตกไฟล์ลงใน OLLAMA_DIR
        subprocess.run(
            f"curl -fsSL {url} | tar -xz -C {OLLAMA_DIR}",
            shell=True,
            check=True,
        )
    # ดึงรายชื่อโมเดล
    result = subprocess.run([OLLAMA_BIN, "list"], capture_output=True, text=True)
    # ถ้าไม่มี gemma3:latest ให้ pull มา
    if "gemma3:latest" not in result.stdout:
        subprocess.run([OLLAMA_BIN, "pull", "gemma3:latest"], check=True)

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
        # ใช้ binary จากโฟลเดอร์ locale รัน gemma3:latest ผ่าน CLI
        proc = subprocess.run(
            [OLLAMA_BIN, "run", "gemma3:latest", "--no-stream", "--prompt", text],
            capture_output=True,
            text=True,
        )
        output = proc.stdout.strip()
        return (output,)

print("📦 prompt_extra_node module loaded")
NODE_CLASS_MAPPINGS = {"PromptExtraNode": PromptExtraNode}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptExtraNode": "Prompt extra"}
print("✅ NODE_CLASS_MAPPINGS defined")
