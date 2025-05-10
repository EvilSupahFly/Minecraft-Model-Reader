import os
import logging
import platformdirs
import sys

# Ensure Amulet's directory is in sys.path so we can import debug_file_access
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import amulet_map_editor.debug_file_access as debug_file_access
    print("minecraft_model_reader/__init__.py: Successfully imported debug_file_access from amulet_map_editor")
except ImportError:
    print("minecraft_model_reader/__init__.py: Failed to import debug_file_access from amulet_map_editor")

# Patch os functions inside minecraft_model_reader
os.path.isfile = debug_file_access.isfile_wrapper
os.path.isdir = debug_file_access.isdir_wrapper
os.path.exists = debug_file_access.exists_wrapper
open = debug_file_access.open_wrapper
os.listdir = debug_file_access.listdir_wrapper

# Initialise default paths. Applications should override these environment variables.
# os.environ.setdefault("DATA_DIR", platformdirs.user_data_dir("AmuletTeam", "AmuletTeam"))
# os.environ.setdefault("CONFIG_DIR", platformdirs.user_config_dir("AmuletTeam", "AmuletTeam"))
os.environ.setdefault("CACHE_DIR", platformdirs.user_cache_dir("AmuletTeam", "AmuletTeam"))
# os.environ.setdefault("LOG_DIR", platformdirs.user_log_dir("AmuletTeam", "AmuletTeam"))

from minecraft_model_reader.api.mesh.block.block_mesh import BlockMesh
from minecraft_model_reader.api.resource_pack import (BaseResourcePack, BaseResourcePackManager,)

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# init a default logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info("minecraft_model_reader/__init__.py: debug_file_access successfully imported and executing.")
logger.info(f"minecraft_model_reader/__init__.py: open is debug_file_access.open_wrapper: {open is debug_file_access.open_wrapper}")
logger.info(f"minecraft_model_reader/__init__.py: os.path.exists is debug_file_access.exists_wrapper: {os.path.exists is debug_file_access.exists_wrapper}")
logger.info(f"minecraft_model_reader/__init__.py: os.listdir is debug_file_access.listdir_wrapper: {os.listdir is debug_file_access.listdir_wrapper}")
logger.info(f"minecraft_model_reader/__init__.py: pathlib.Path.open is debug_file_access.path_open_wrapper: {debug_file_access.pathlib.Path.open is debug_file_access.path_open_wrapper}")
logger.info(f"minecraft_model_reader/__init__.py: os.path.isfile is debug_file_access.isfile_wrapper: {os.path.isfile is debug_file_access.isfile_wrapper}")
logger.info(f"minecraft_model_reader/__init__.py: os.path.isdir is debug_file_access.isdir_wrapper: {os.path.isdir is debug_file_access.isdir_wrapper}")
