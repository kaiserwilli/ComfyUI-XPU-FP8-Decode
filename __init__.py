"""ComfyUI custom-node registration for the text generation optimization."""
import logging
from .engine import GenerationClass, generate
from .settings import TAG

GenerationClass.generate = generate
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
logging.info("%s: cleaned extension loaded", TAG)
