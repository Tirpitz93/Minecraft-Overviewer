import os
import json
from collections import defaultdict
from functools import lru_cache
import moderngl as mgl
import numpy as np
from math import sin, cos, tan, asin, pi

from PIL import Image
import logging

logger = logging.getLogger(__name__)
START_BLOCK_ID = 20000
BLOCK_LIST = [
    "dirt",
    "cyan_stained_glass",
    "gray_concrete_powder",
    "red_terracotta",
    "quartz_block",
    "beehive",
    "birch_leaves",
    "gray_concrete",
    "brown_glazed_terracotta",
    "magma_block",
    "hay_block",
    "oak_log",
    "end_stone_bricks",
    "iron_ore",
    "emerald_block",
    "light_blue_concrete_powder",
    "black_concrete_powder",
    "jigsaw",
    "dispenser",
    "green_terracotta",
    "dead_brain_coral_block",
    "red_concrete",
    "mossy_cobblestone",
    "barrel",
    "repeating_command_block",
    "stripped_jungle_log",
    "magenta_concrete",
    "gray_wool",
    "diamond_ore",
    "orange_stained_glass",
    "white_wool",
    "blue_ice",
    "terracotta",
    "redstone_lamp",
    "dark_oak_planks",
    "wither_skeleton_wall_skull",
    "polished_andesite",
    "ice",
    "purple_glazed_terracotta",
    "jungle_log",
    "prismarine",
    "light_gray_concrete",
    "red_sandstone",
    "orange_concrete_powder",
    "coarse_dirt",
    "gold_block",
    "light_blue_terracotta",
    "wet_sponge",
    "white_glazed_terracotta",
    "lime_terracotta",
    "white_concrete",
    "green_wool",
    "birch_planks",
    "purpur_block",
    "cracked_stone_bricks",
    "lapis_block",
    "iron_block",
    "magenta_concrete_powder",
    "black_stained_glass",
    "chiseled_red_sandstone",
    "spruce_planks",
    "coal_ore",
    "red_nether_bricks",
    "crafting_table",
    "stone_bricks",
    "black_concrete",
    "stripped_jungle_wood",
    "smoker",
    "pink_wool",
    "diamond_block",
    "black_glazed_terracotta",
    "chiseled_sandstone",
    "dark_oak_wood",
    "mycelium",
    "gray_glazed_terracotta",
    "purpur_pillar",
    "lime_wool",
    "cyan_concrete_powder",
    "infested_cracked_stone_bricks",
    "stripped_spruce_log",
    "yellow_wool",
    "dead_bubble_coral_block",
    "oak_planks",
    "white_concrete_powder",
    "dark_oak_leaves",
    "emerald_ore",
    "jack_o_lantern",
    "pink_terracotta",
    "podzol",
    "nether_quartz_ore",
    "andesite",
    "end_stone",
    "chiseled_stone_bricks",
    "sponge",
    "stripped_acacia_wood",
    "furnace",
    "brown_concrete",
    "cut_sandstone",
    "red_sand",
    "light_blue_wool",
    "cyan_concrete",
    "infested_mossy_stone_bricks",
    "light_blue_stained_glass",
    "cyan_glazed_terracotta",
    "acacia_planks",
    "blast_furnace",
    "netherrack",
    "nether_bricks",
    "piston",
    "infested_stone_bricks",
    "fire_coral_block",
    "skeleton_wall_skull",
    "snow_block",
    "stripped_oak_wood",
    "lime_concrete_powder",
    "light_gray_wool",
    "gray_stained_glass",
    "lime_concrete",
    "quartz_pillar",
    "dead_tube_coral_block",
    "acacia_leaves",
    "cartography_table",
    "red_glazed_terracotta",
    "dropper",
    "birch_log",
    "red_stained_glass",
    "diorite",
    "smithing_table",
    "glowstone",
    "melon",
    "wither_skeleton_skull",
    "yellow_terracotta",
    "light_gray_stained_glass",
    "prismarine_bricks",
    "cyan_wool",
    "jukebox",
    "white_terracotta",
    "mossy_stone_bricks",
    "green_concrete",
    "orange_concrete",
    "pink_concrete",
    "blue_concrete",
    "brown_concrete_powder",
    "white_stained_glass",
    "sand",
    "stripped_dark_oak_wood",
    "yellow_stained_glass",
    "cobblestone",
    "chain_command_block",
    "lapis_ore",
    "acacia_log",
    "spruce_wood",
    "spruce_leaves",
    "smooth_stone",
    "oak_wood",
    "oak_leaves",
    "purple_concrete_powder",
    "gray_terracotta",
    "yellow_concrete_powder",
    "note_block",
    "magenta_stained_glass",
    "smooth_quartz",
    "light_gray_glazed_terracotta",
    "magenta_glazed_terracotta",
    "infested_cobblestone",
    "clay",
    "horn_coral_block",
    "glass",
    "stripped_spruce_wood",
    "lime_glazed_terracotta",
    "purple_wool",
    "smooth_red_sandstone",
    "light_blue_concrete",
    "obsidian",
    "brown_wool",
    "blue_stained_glass",
    "light_blue_glazed_terracotta",
    "red_wool",
    "jungle_planks",
    "purple_terracotta",
    "gravel",
    "blue_concrete_powder",
    "light_gray_terracotta",
    "cut_red_sandstone",
    "dark_prismarine",
    "black_terracotta",
    "loom",
    "yellow_concrete",
    "light_gray_concrete_powder",
    "soul_sand",
    "pink_concrete_powder",
    "command_block",
    "sea_lantern",
    "spruce_log",
    "skeleton_skull",
    "bubble_coral_block",
    "lime_stained_glass",
    "purple_stained_glass",
    "jungle_leaves",
    "sandstone",
    "fletching_table",
    "green_stained_glass",
    "polished_granite",
    "dead_horn_coral_block",
    "stripped_birch_wood",
    "bee_nest",
    "brown_stained_glass",
    "yellow_glazed_terracotta",
    "cyan_terracotta",
    "green_concrete_powder",
    "pink_glazed_terracotta",
    "black_wool",
    "purple_concrete",
    "bone_block",
    "blue_terracotta",
    "stripped_dark_oak_log",
    "green_glazed_terracotta",
    "redstone_ore",
    "blue_wool",
    "dark_oak_log",
    "orange_terracotta",
    "packed_ice",
    "chiseled_quartz_block",
    "orange_glazed_terracotta",
    "bricks",
    "honeycomb_block",
    "structure_block",
    "sticky_piston",
    "acacia_wood",
    "bookshelf",
    "pumpkin",
    "stripped_oak_log",
    "stripped_acacia_log",
    "blue_glazed_terracotta",
    "frosted_ice",
    "redstone_block",
    "orange_wool",
    "brain_coral_block",
    "gold_ore",
    "nether_wart_block",
    "jungle_wood",
    "coal_block",
    "magenta_terracotta",
    "smooth_sandstone",
    "tube_coral_block",
    "stripped_birch_log",
    "polished_diorite",
    "magenta_wool",
    "spawner",
    "dead_fire_coral_block",
    "pink_stained_glass",
    "carved_pumpkin",
    "red_concrete_powder",
    "granite",
    "birch_wood",
    "brown_terracotta",
    "infested_chiseled_stone_bricks",
    "tnt",
]

################################################################
# Constants and helper methods for the BlockRenderer
# Placed here in order to not need self.
################################################################
N = "north"
E = "east"
S = "south"
W = "west"
U = "up"
D = "down"


def rot(image, degree):
    if image is None:
        return None
    return image.rotate(degree)


def flip(image, arg1):
    if image is not None:
        return image.transpose(arg1)
    else:
        return None


################################################################
# Simple function for loading .obj files
################################################################
def load_obj(ctx, render_program, path):
    """
    This method loads a .obj file and creates a model that can be rendered using moderngl.

    Wavefont (.obj) files reference verticies, uvs and normals for each vertex with different indicies.
    ModernGL does not support different indicies for verticies, uvs and normals.

    In addition to de-referencing the verticies, uvs and normals they are combined into a single VertexBuffer.

    Finally a VertexArray is created, that can be rendered using
    return_value.render(mode=mgl.TRIANGLES)
    """
    # Store all verticies, uvs, normals and faces for later processing
    raw_verticies = []
    raw_uvs = []
    raw_normals = []
    raw_faces = []

    # Read data from the file and store it into the arrays above
    with open(path, "r") as fp:
        for line in fp.readlines():
            line = line.strip()
            if line[0] == '#':
                continue
            args = tuple(line.split(' ')[1:])
            if line.startswith('v '):
                raw_verticies.append(args)
            elif line.startswith('vt '):
                raw_uvs.append(args)
            elif line.startswith('vn '):
                raw_normals.append(args)
            elif line.startswith('f '):
                raw_faces.append((x.split('/') for x in args))
            else:
                pass

    # Combine all arrays and do the de-referencing
    data = np.array([
        value
        for face in raw_faces
        for vertex in face
        for value_list in [         # Each vertex consists of a position, normal and a uv
            raw_verticies[int(vertex[0])-1],
            raw_normals[int(vertex[2])-1],
            raw_uvs[int(vertex[1])-1],
        ]
        for value in value_list     # Combine all small arrays into a single large one
    ], dtype="f4")

    # By reshaping the array all values can be read more easily
    # print(data.reshape((data.size // 8, 8)))

    # Create a buffer containing the data
    cube_vbo = ctx.buffer(data.tobytes())
    # Create a VertexArray bound to the buffer and the render_program
    return ctx.simple_vertex_array(render_program, cube_vbo, "in_vert", "in_normal", "in_texcoord_0")


################################################################
# Main Code for Block Rendering
################################################################
class BlockRenderer(object):
    # Paths in the jar file
    BLOCKSTATES_DIR = "assets/minecraft/blockstates"
    MODELS_DIR = "assets/minecraft/models"
    TEXTURES_DIR = "assets/minecraft/textures"

    # Storage for finding the data value
    data_map = defaultdict(list)

    def __init__(self, textures, *, block_list=BLOCK_LIST, start_block_id: int=1, resolution: int=24,
                 vertex_shader: str="overviewer_core/rendering/default.vert",
                 fragment_shader: str="overviewer_core/rendering/default.frag",
                 projection_matrix=None):
        # Not direclty related to rendering
        self.textures = textures
        self.start_block_id = start_block_id
        self.block_list = block_list

        # Settings for rendering
        self.resolution = resolution

        # Configure rendering
        self.ctx, self.fbo, self.cube_model = self.setup_rendering(vertex_shader, fragment_shader, projection_matrix)

    def setup_rendering(self, vertex_shader, fragment_shader, projection_matrix=None):
        # Read shader source-code
        with open(vertex_shader) as fp:
            vertex_shader_src = fp.read()
        with open(fragment_shader) as fp:
            fragment_shader_src = fp.read()

        # Setup for rendering
        ctx = mgl.create_context(
            standalone=True,
            backend='egl',
            libgl='libGL.so.1',
            libegl='libEGL.so.1',
        )
        # DEPTH_TEST to calculate which face is visible, CULL_FACE to hide the backside of each face
        ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
        # Create a framebuffer to render into
        fbo = ctx.simple_framebuffer((self.resolution, self.resolution), components=4)
        fbo.use()
        # Compile the shaders
        render_program = ctx.program(vertex_shader=vertex_shader_src, fragment_shader=fragment_shader_src)

        # Load cube.obj to get a model to render (ArrayBuffer)
        cube_vao = load_obj(ctx, render_program, "overviewer_core/rendering/cube.obj")

        # If no projection_matrix is given use the kind-of isometric view used by Overviewer
        if projection_matrix is None:
            projection_matrix = self.calculate_projection_matrix()

        # Load and use a texture
        # TODO: Replace this by a Texturemap and adjust the shaders
        img = self.load_img("block/oak_planks")
        texture = ctx.texture(img.size, 4, img.tobytes())
        texture.filter = (mgl.NEAREST, mgl.NEAREST)     # Use the nearest pixel instead of lineary interpolating
        texture.use()

        # Set the "uniform" values the shaders require
        render_program["Mvp"].write(projection_matrix.astype('f4').tobytes())

        return ctx, fbo, cube_vao

    def calculate_projection_matrix(self):
        # Orthographic view matrix
        top = .8165
        bot = -.8165
        left = -.7073
        right = .7073
        near = -10
        far = 10
        projection_matrix = np.array([
            [2 / (right-left), 0, 0, -(right+left)/(right-left)],
            [0, 2/(top-bot), 0, -(top+bot)/(top-bot)],
            [0, 0, -2/(far-near), -(far+near)/(far-near)],
            [0, 0, 0, 1],
        ], dtype="f4")

        # Rotation matricies from Wikipedia: https://en.wikipedia.org/wiki/Rotation_matrix
        # Alpha values from Wikipedia: https://en.wikipedia.org/wiki/Isometric_projection#Mathematics
        alpha = asin(tan(pi / 6))
        print(alpha)
        s = sin(alpha)
        c = cos(alpha)
        rot_x = np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])

        alpha = pi / 4
        s = sin(alpha)
        c = cos(alpha)
        rot_y = np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])

        transform = np.matmul(rot_x, rot_y)
        view_matrix = np.linalg.inv(transform)

        # roty*rotx is the normal isometric view, the scale makes it fit into a square after applying the isometric view
        # The transform matrix must be inverted and combined with the projection_matrix to get the VP matrix
        return np.matmul(view_matrix, projection_matrix)

    ################################################################
    # Loading files
    ################################################################
    def load_file(self, path:str, name:str, ext:str):
        if ":" in name:
            return self.textures.find_file("{0}/{1}{2}".format(path,name.split(":")[1],ext))
        else:
            return self.textures.find_file("{0}/{1}{2}".format(path,name,ext))

    def load_json(self, name: str, directory: str) -> dict:
        with self.load_file(directory, name, ".json") as f:
            return json.load(f)

    def load_blockstates(self, name: str) -> dict:
        return self.load_json(name, self.BLOCKSTATES_DIR)

    def load_model(self, name: str) -> dict:
        return self.load_json(name, self.MODELS_DIR)

    @lru_cache()
    def load_img(self, texture_name):
        with self.load_file(self.TEXTURES_DIR, texture_name, ".png") as f:
            return Image.open(f).convert("RGBA")

    ################################################################
    # Model file parsing
    ################################################################
    @staticmethod
    def combine_textures_fields(textures_field: dict, parent_textures_field: dict) -> dict:
        return {
            **{
                key: textures_field.get(value[1:], value) if value[0] == '#' else value
                for key, value in parent_textures_field.items()
            },
            **textures_field
        }

    def load_and_combine_model(self, name):
        data = self.load_model(name)
        if "parent" in data:
            # Load the JSON from the parent
            parent_data = self.load_and_combine_model(data["parent"])
            elements_field = data.get("elements", parent_data.get("elements"))
            textures_field = self.combine_textures_fields(data.get("textures", {}), parent_data.get("textures", {}))
        else:
            elements_field = data.get("elements")
            textures_field = data.get("textures", {})

        return {
            "textures": textures_field,
            "elements": elements_field,
        }

    def get_max_nbt_count(self, name: str) -> int:
        data = self.load_blockstates(name)
        return len(data.get("variants", []))

    ################################################################
    # Render methods
    ################################################################
    def render_vertex_array(self, vertex_array: mgl.VertexArray, pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        # Write uniform values and render the vertex_array
        vertex_array.program["pos"].write(np.array(pos, dtype="f4"))
        vertex_array.program["scale"].write(np.array(scale, dtype="f4"))
        vertex_array.render()

    def render_element(self, element, rotation_x_axis, rotation_y_axis, uvlock):
        # Convert the two cube corners into postion, and scale
        pos = tuple((t + f) / 32 - .5 for f, t in zip(element["from"], element["to"]))
        rot = (0, 0, 0)     # Not implemented yet
        scale = tuple((t - f) / 16 for f, t in zip(element["from"], element["to"]))

        # Render the cube
        self.render_vertex_array(self.cube_model, pos, rot, scale)

    def render_model(self, data: dict, rotation_x_axis, rotation_y_axis, uvlock):
        """
        This method is currently only using existing draw methods. Because of that only full size blocks can be created.

        Assumption:
        - The bottom texture is pointing in the reverse direction of the top texture
        Reason: models/blocks/template_glaced_terracotta.json does not rotate the top or bottom faces
        ==> This Indicates the default rotation

        Limitations:
        - Only full blocks
        - Rotation is not supported
        - UV coordinates must be [0, 0, 16, 16] (can be implemented bt isn't yet)
        - tintcolor
        """
        # Check if there are errors and the block can't be rendered correctly because of limitations listed above
        if data["elements"] is None or len(data["elements"]) == 0:
            raise NotImplementedError("Only blocks with 'elements' are supported.")

        # Check if the blocks is currently supported
        for part in data["elements"]:
            if "rotation" in part:
                raise NotImplementedError("Rotated Elements are not supported")

            for face_name, definition in part["faces"].items():
                if "uv" in definition and definition["uv"] != [0, 0, 16, 16]:
                    # raise NotImplementedError("Only elements with UV [0, 0, 16, 16] are supported")
                    pass

        # Clear the renderbuffer to start a new image
        self.ctx.clear()

        # Render the parts in the order they are in the file.
        # Reason: Required for correct rendering of e.g. the grass block
        for part in data["elements"]:
            # Render a single cube
            self.render_element(part, rotation_x_axis, rotation_y_axis, uvlock)

        # Read the data from the framebuffer and return it as a PIL.Image
        img = Image.frombytes("RGBA", (self.resolution, self.resolution), self.fbo.read(components=4))
        return img.transpose(Image.FLIP_TOP_BOTTOM)

    def render_blockstate_entry(self, data: dict) -> Image:
        # model, x, y, uvlock, weight
        return self.render_model(
            data=self.load_and_combine_model(data["model"]),
            rotation_x_axis=data.get("x", 0),  # Increments of 90°
            rotation_y_axis=data.get("y", 0),  # Increments of 90°
            uvlock=data.get("uvlock", False)
        ), data.get("weight", 1)

    def render_blockstates(self, data: dict) -> (str, []):
        if "variants" in data:
            for nbt_condition, variant in data["variants"].items():
                yield (nbt_condition, [
                    self.render_blockstate_entry(v)
                    for v in (variant if type(variant) == list else (variant,))
                ])
        else:
            raise NotImplementedError("Multipart is not supported. Only blocks using 'variants' are.")

    ################################################################
    # NBT to Data conversion
    ################################################################
    @staticmethod
    def store_nbt_as_int(name, nbt_condition, blockid, data_value):
        compare_set = {x.split('=')[0] for x in nbt_condition.split(',') if x != ""}
        BlockRenderer.data_map["minecraft:%s" % name].append((compare_set, (blockid, data_value)))

    @staticmethod
    def get_nbt_as_int(key: str, properties: dict):
        entry = BlockRenderer.data_map.get(key)
        if entry is None:
            return None, None

        if properties is None:
            properties = {}

        for stored_set, data_value in entry:
            for x in properties.items():
                if "%s=%s" % x not in stored_set:
                    break
            else:
                return data_value
        # If nothing matches return something (better than drawing nothing)
        # This should only happen if the BlockRenderer can't render a variant
        if len(entry) > 0:
            return entry[0][1]
        else:
            # No valid variant found (probably won't ever happen because no entry would be created)
            return None, None

    ################################################################
    # Genrators and methods for rendering multiple blocks
    ################################################################
    def iter_blocks(self, name_list: list, ignore_unsupported_blocks=True):
        for block_index, block_name in enumerate(name_list):
            try:
                for nbt_index, (nbt_condition, variants) in enumerate(
                        self.render_blockstates(self.load_blockstates(name=block_name))
                ):
                    yield block_index, block_name, nbt_index, nbt_condition, variants
            except NotImplementedError as e:
                if not ignore_unsupported_blocks:
                    raise e

    def iter_all_blocks(self, ignore_unsupported_blocks=True):
        # TODO: Getting the find_file_local_path from textures is cheating and only works if the jar file is extracted
        blockstates_dir = self.textures.find_file_local_path + "/assets/minecraft/blockstates"
        logger.debug("Searching for blockstates in " + blockstates_dir)
        return self.iter_blocks([
            fn.split('.', 1)[0]
            for _, _, files in os.walk(blockstates_dir)
            for fn in files
            if fn.split('.', 1)[1] == "json"
        ], ignore_unsupported_blocks=ignore_unsupported_blocks)

    def get_max_size(self) -> (int, int):
        blockid_count = len(self.block_list)
        data_count = max(self.get_max_nbt_count(name) for name in self.block_list)
        return blockid_count + self.start_block_id, data_count

    def iter_for_generate(self):
        for block_index, block_name, nbt_index, nbt_condition, variants in self.iter_all_blocks():
            if len(variants) >= 1:
                logger.debug("Block found: {0} -> {1}:{2}".format(block_name, block_index, nbt_index))
                BlockRenderer.store_nbt_as_int(block_name, nbt_condition, block_index + self.start_block_id, nbt_index)
                yield (block_index + self.start_block_id, nbt_index), variants[0][0]

