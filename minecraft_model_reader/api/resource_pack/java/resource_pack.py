import os, json
import logging

from minecraft_model_reader.api.resource_pack.base import BaseResourcePack
log = logging.getLogger(__name__)

# Define TRACE level (lower than DEBUG)
TRACE_LEVEL = 1
logging.addLevelName(TRACE_LEVEL, "TRACE")

# Add a trace method to the logging class
def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)

# Attach the trace method to the Logger class
logging.Logger.trace = trace
logging.TRACE_LEVEL = TRACE_LEVEL
logging.basicConfig(level=logging.TRACE_LEVEL)
log.setLevel(TRACE_LEVEL)

class JavaResourcePack(BaseResourcePack):
    """A class to hold the bare bones information about the resource pack.
    Holds the pack format, description and if the pack is valid.
    This information can be used in a viewer to display the packs to the user."""

    def __init__(self, resource_pack_path: str):
        log.trace(f"Initializing JavaResourcePack with path: {resource_pack_path}")
        super().__init__(resource_pack_path)
        meta_path = os.path.join(resource_pack_path, "pack.mcmeta")
        self._pack_format = 0
        if os.path.isfile(meta_path):
            log.trace(f"Found pack.mcmeta: {meta_path}")
            try:
                with open(meta_path) as f:
                    pack_mcmeta = json.load(f)
                log.trace(f"pack.mcmeta contents: {pack_mcmeta}")
            except json.JSONDecodeError as e:
                log.error(f"Error decoding pack.mcmeta at {meta_path}: {e}")
                pass
            else:
                if "pack" in pack_mcmeta:
                    if "description" in pack_mcmeta["pack"]:
                        self._pack_description = str(pack_mcmeta["pack"]["description"])
                        log.trace(f"Pack description: {self._pack_description}")
                    if "pack_format" in pack_mcmeta["pack"]:
                        self._pack_format = pack_mcmeta["pack"]["pack_format"]
                        log.trace(f"Pack format: {self._pack_format}")
                        self._valid_pack = True

        pack_icon_path = os.path.join(resource_pack_path, "pack.png")
        if os.path.isfile(pack_icon_path):
            self._pack_icon = pack_icon_path
            log.trace(f"Pack icon found at: {pack_icon_path}")

        texture_root = os.path.join(resource_pack_path, "assets/minecraft/textures/block")
        log.trace(f"__init__: Checking for textures in {texture_root}")
        # Check if water/lava textures exist in the pack
        for texture in ["water*.png", "lava*.png"]:            
            texture_path = os.path.join(texture_root, texture)
            if os.path.isfile(texture_path):
                log.trace(f"__init__: Texture (texture): {texture}")
                log.trace(f"__init__: Resource Pack Path (resource_pack_path): {resource_pack_path}")
                log.trace(f"__init__: Found texture (texture_path): {texture_path}")
            else:
                log.warning(f"__init__: Texture (texture): {texture}")
                log.warning(f"__init__: Resource Pack Path (resource_pack_path): {resource_pack_path}")
                log.warning(f"__init__: Missing texture: {texture_path}")

    @staticmethod
    def is_valid(pack_path: str) -> bool:
        valid = os.path.isfile(os.path.join(pack_path, "pack.mcmeta"))
        log.trace(f"Checking if resource pack is valid: {valid}")
        return valid

    def __repr__(self) -> str:
        return f"JavaResourcePack({self._root_dir})"

    @property
    def pack_format(self) -> int:
        """int - pack format number"""
        return self._pack_format
