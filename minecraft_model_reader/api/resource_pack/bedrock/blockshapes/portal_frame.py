from typing import Tuple

from minecraft_model_reader.api.resource_pack.bedrock.blockshapes.partial_block import PartialBlock
from minecraft_model_reader.api import Block


class PortalFrame(PartialBlock):
    @property
    def blockshape(self) -> str:
        return "portal_frame"

    def bounds(self, block: Block) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        return (0, 1), (0, 13/16), (0, 1)

    @property
    def do_not_cull(self) -> Tuple[bool, bool, bool, bool, bool, bool]:
        return False, True, False, False, False, False


BlockShape = PortalFrame()
