import logging
from typing import Iterable, Union

from minecraft_model_reader.api.resource_pack.base import (BaseResourcePack, BaseResourcePackManager,)
from minecraft_model_reader.api.resource_pack.java import (JavaResourcePack, JavaResourcePackManager,)
from minecraft_model_reader.api.resource_pack.bedrock import (BedrockResourcePack, BedrockResourcePackManager,)
from .unknown_resource_pack import UnknownResourcePack

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

def load_resource_pack(resource_pack_path: str) -> BaseResourcePack:
    if JavaResourcePack.is_valid(resource_pack_path):
        log.trace(f"load_resource_pack: Valid Java Pack found: {resource_pack_path}")
        return JavaResourcePack(resource_pack_path)
    elif BedrockResourcePack.is_valid(resource_pack_path):
        log.trace(f"load_resource_pack: Valid Bedrock Pack found: {resource_pack_path}")
        return BedrockResourcePack(resource_pack_path)
    else:
        log.trace(f"load_resource_pack: Unknown Pack found: {resource_pack_path}")
        return UnknownResourcePack(resource_pack_path)

def load_resource_pack_manager(resource_packs: Iterable[Union[str, BaseResourcePack]], load: bool = True) -> BaseResourcePackManager:
    log.trace(f"load_resource_pack_manager: Initial resource_packs = {resource_packs}")
    resource_packs_out: list[BaseResourcePack] = []
    for resource_pack in resource_packs:
        if isinstance(resource_pack, str):
            resource_pack = load_resource_pack(resource_pack)
        log.trace(f"Checking resource_pack: {resource_pack}, vars: {vars(resource_pack)}")
        if (not isinstance(resource_pack, UnknownResourcePack) and resource_pack.valid_pack):
            if resource_packs_out:
                if isinstance(resource_pack, resource_packs_out[0].__class__):
                    resource_packs_out.append(resource_pack)
            else:
                resource_packs_out.append(resource_pack)

    resource_packs = resource_packs_out
    log.trace(f"load_resource_pack_manager: Filtered resource_packs_out = {resource_packs_out}")
    if resource_packs:
        if isinstance(resource_packs[0], JavaResourcePack):
            return JavaResourcePackManager([pack for pack in resource_packs if isinstance(pack, JavaResourcePack)], load,)
        elif isinstance(resource_packs[0], BedrockResourcePack):
            return BedrockResourcePackManager([pack for pack in resource_packs if isinstance(pack, BedrockResourcePack)],load,)

    raise NotImplementedError
    # return UnknownResourcePackManager()
