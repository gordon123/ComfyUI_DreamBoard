# __init__.py
# รวม node classes ทั้งหมดไว้ใน package นี้

from .storyboard_node import StoryboardNode
from .prompt_extra_node import PromptExtraNode

# Mapping class names to their implementations
NODE_CLASS_MAPPINGS = {
    "StoryboardNode": StoryboardNode,
    "PromptExtraNode": PromptExtraNode,
}

# Mapping class names to display names shown in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "StoryboardNode": "🎬 Storyboard Image → Prompt",
    "PromptExtraNode": "Prompt extra",
}
