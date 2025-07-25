import os
import json
import traceback
import logging

log = logging.getLogger(__name__)

from minecraft_model_reader.api.resource_pack.base import BaseResourcePack


class JavaResourcePack(BaseResourcePack):
    """A class to hold the bare bones information about the resource pack.
    Holds the pack format, description and if the pack is valid.
    This information can be used in a viewer to display the packs to the user."""

    def __init__(self, resource_pack_path: str):
        log.info(f"[minecraft_model_reader/api/resource_pack/java/resource_pack.py: JavaResourcePack] Init with path: {resource_pack_path}")
        log.info(f"[minecraft_model_reader/api/resource_pack/java/resource_pack.py: JavaResourcePack] Contents of path: {os.listdir(resource_pack_path) if os.path.isdir(resource_pack_path) else 'Not a directory'}")
        traceback.print_stack()
        super().__init__(resource_pack_path)
        meta_path = os.path.join(resource_pack_path, "pack.mcmeta")
        self._pack_format = 0
        if os.path.isfile(meta_path):
            log.info(f"[minecraft_model_reader/api/resource_pack/java/resource_pack.py: JavaResourcePack] Found pack.mcmeta at: {meta_path}")
            try:
                with open(meta_path) as f:
                    pack_mcmeta = json.load(f)
                log.info(f"[minecraft_model_reader/api/resource_pack/java/resource_pack.py: JavaResourcePack] pack.mcmeta content: {pack_mcmeta}")
            except json.JSONDecodeError:
                pass
            else:
                if "pack" in pack_mcmeta:
                    if "description" in pack_mcmeta["pack"]:
                        self._pack_description = str(pack_mcmeta["pack"]["description"])
                    if "pack_format" in pack_mcmeta["pack"]:
                        self._pack_format = pack_mcmeta["pack"]["pack_format"]
                        self._valid_pack = True

        pack_icon_path = os.path.join(resource_pack_path, "pack.png")
        if os.path.isfile(pack_icon_path):
            self._pack_icon = pack_icon_path

    @staticmethod
    def is_valid(pack_path: str) -> bool:
        return os.path.isfile(os.path.join(pack_path, "pack.mcmeta"))

    def __repr__(self) -> str:
        return f"JavaResourcePack({self._root_dir})"

    @property
    def pack_format(self) -> int:
        """int - pack format number"""
        return self._pack_format
