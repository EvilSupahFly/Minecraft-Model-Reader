from typing import TYPE_CHECKING

from .block_mesh import BlockMesh
from .cube import get_unit_cube

if TYPE_CHECKING:
    from minecraft_model_reader.api.resource_pack.base import BaseResourcePackManager


#def get_missing_block(resource_pack: "BaseResourcePackManager") -> BlockMesh:
#    texture_path = resource_pack.get_texture_path("minecraft", "missing_no")
#    return get_unit_cube(
#        texture_path,
#        texture_path,
#        texture_path,
#        texture_path,
#        texture_path,
#        texture_path,
#    )

def get_missing_block(resource_pack: "BaseResourcePackManager") -> BlockMesh:
    """
    Retrieves a BlockMesh representing a missing block texture.

    Parameters:
    resource_pack (BaseResourcePackManager): The resource pack manager to fetch textures from.

    Returns:
    BlockMesh: A BlockMesh object with the missing block texture applied.
    """
    # Fetch the texture path for the missing block
    texture_path = resource_pack.get_texture_path("minecraft", "missing_no")
    print(f"Texture path retrieved: {texture_path}")

    # Create a unit cube with the missing block texture applied to all faces
    block_mesh = get_unit_cube(texture_path, texture_path, texture_path, texture_path, texture_path, texture_path)
    print(f"BlockMesh created with texture path: {texture_path}")

    return block_mesh