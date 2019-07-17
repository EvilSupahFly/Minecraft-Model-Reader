from typing import Tuple, Dict, Union, List
import itertools
import pyglet
import numpy

import minecraft_model_reader

from amulet.api import paths
from amulet.api.block import Block
paths.FORMATS_DIR = r"./amulet/formats"
paths.DEFINITIONS_DIR = r"./amulet/version_definitions"
from amulet import world_loader

import minecraft_model_reader

cull_offset_dict = {'down': (0,-1,0), 'up': (0,1,0), 'north': (0,0,-1), 'east': (1,0,0), 'south': (0,0,1), 'west': (-1,0,0)}

class RenderChunk:
	def __init__(self, batch, world, resource_pack, cx, cz):
		self.batch = batch
		self.cx = cx
		self.cz = cz
		blocks = world.get_chunk(cx, cz).blocks
		vert_list = []
		face_list = []
		vert_count = 0
		block_dict = {}
		for x, y, z in itertools.product(range(16), range(256), range(16)):
			block = blocks[x, y, z]
			block_dict.setdefault(block, [])
			block_dict[block].append((x, y, z))

		for block_temp_id, block_locations in block_dict.items():
			block = world.block_manager[
				block_temp_id
			]
			model: minecraft_model_reader.MinecraftMesh = resource_pack.get_model(
				block
			)
			block_count = len(block_locations)
			block_offsets = numpy.array(block_locations) + (cx*16, 0, cz*16)
			for cull_dir in model.faces.keys():
				verts = model.verts[cull_dir][:, :3]
				mini_vert_count = len(verts)
				vert_list += list(numpy.tile(verts.ravel(), block_count).ravel() + numpy.repeat(block_offsets, mini_vert_count, axis=0).ravel())
				faces = model.faces[cull_dir][:, :-1]
				face_list += list(numpy.tile(faces.ravel(), block_count).ravel() + numpy.repeat(numpy.arange(vert_count, vert_count + mini_vert_count * block_count, mini_vert_count), faces.size))
				vert_count += mini_vert_count * block_count

		self.batch.add_indexed(
			int(len(vert_list)/3),
			pyglet.gl.GL_TRIANGLES,
			None,
			face_list,
			('v3f', vert_list)
		)


class RenderWorld:
	def __init__(self, batch, world_path: str, resource_packs: Union[str, List[str]]):
		self.batch = batch
		self.world = world_loader.load_world(world_path)
		self.chunks: Dict[Tuple[int, int], RenderChunk] = {}

		self.render_distance = 1
		self.busy = False

		# Load the resource pack
		if isinstance(resource_packs, str):
			resource_packs = minecraft_model_reader.JavaRP(resource_packs)
		elif isinstance(resource_packs, list):
			resource_packs = [minecraft_model_reader.JavaRP(rp) for rp in resource_packs]
		else:
			raise Exception('resource_pack must be a string or list of strings')
		self.resource_pack = minecraft_model_reader.JavaRPHandler(resource_packs)
		self.textures = {}

	def get_texture(self, namespace_and_path: Tuple[str, str]):
		if namespace_and_path not in self.textures:
			abs_texture_path = self.resource_pack.get_texture(*namespace_and_path)
			self.textures[namespace_and_path] = pyglet.image.load(abs_texture_path)

		return self.textures[namespace_and_path]

	def draw(self):
		self.batch.draw()

	def update(self, x, z):
		if not self.busy:
			self.busy = True
			cx = int(x) >> 4
			cz = int(z) >> 4
			chunk = next(
				(
					chunk for chunk in sorted(
						itertools.product(
							range(
								cx-self.render_distance,
								cx+self.render_distance
							),
							range(
								cz - self.render_distance,
								cz + self.render_distance
							)
						),
						key=lambda chunk_coords: chunk_coords[0]**2 + chunk_coords[1] ** 2
					) if chunk not in self.chunks
				),
				None
			)
			if chunk is None:  # All the chunks in the render distance have been loaded
				self.busy = False
			else:
				cx, cz = chunk
				self.chunks[chunk] = RenderChunk(self.batch, self.world, self.resource_pack, cx, cz)

			self.busy = False

		# unload chunks outside the render distance
		# for cx, cz in self.chunks.keys():
		# 	if cx, cz