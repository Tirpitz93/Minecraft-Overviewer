import ctypes
import math
import os
import typing
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Tuple

import moderngl as mgl
import numpy as np
from math import sin, cos, tan, asin, pi

from overviewer_core.util import get_program_path
from PIL import Image
import logging

logger = logging.getLogger(__name__)
from overviewer_core.asset_loader import AssetLoader

START_BLOCK_ID = 20000


################################################################
# Simple function for loading .obj files
################################################################


class NoElementDataInDefinition(Exception):
    pass


class NoContextCreated(Exception):
    def __init__(self, exceptions):
        self.exceptions = exceptions


################################################################
# Main Code for Block Rendering
################################################################
class MGlRenderer(object):
    # DEFAULT_LIGHT_VECTOR = (-0.9, 1, 0.8)
    DEFAULT_LIGHT_VECTOR = (-.8, 1, .7)

    def __init__(self, vertex_shader: str, fragment_shader: str, projection_matrix: np.array, texture_list: iter,
                 light_vector=DEFAULT_LIGHT_VECTOR, output_resolution=24, texture_resolution=16):
        self.output_resolution = output_resolution

        # Read shader source-code
        with open(os.path.join(get_program_path(), vertex_shader)) as fp:
            vertex_shader_src = fp.read()
        with open(os.path.join(get_program_path(), fragment_shader)) as fp:
            fragment_shader_src = fp.read()

        # Calculate the projection matrix if not given
        if projection_matrix is None:
            projection_matrix = self.calculate_projection_matrix()

        # Setup rendering
        self.ctx, self.fbo, self.cube_model = self.setup_rendering(
            vertex_shader_src, fragment_shader_src, texture_list, projection_matrix,
            light_vector, output_resolution, texture_resolution
        )

    def setup_rendering(self, vertex_shader_src: str, fragment_shader_src: str, texture_list: iter,
                        projection_matrix: np.array, light_vector: tuple, output_resolution: int,
                        texture_resolution: int):
        # Setup for rendering
        ctx = self._create_context()
        logger.debug("moderngl context info: %s", str(ctx.info))

        # DEPTH_TEST to calculate which face is visible, CULL_FACE to hide the backside of each face
        ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
        # ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        # Create a framebuffer to render into
        fbo = ctx.simple_framebuffer((output_resolution, output_resolution), components=4)
        fbo.use()
        # Compile the shaders
        render_program = ctx.program(vertex_shader=vertex_shader_src, fragment_shader=fragment_shader_src)

        # Load cube.obj to get a model to render (ArrayBuffer)
        cube_vao = self.load_obj(ctx, render_program, "overviewer_core/rendering/cube.obj")

        # Load all textures into a single TextureArray
        # All textures have to be of size mc_texture_size*mc_texture_size
        texture_atlas, atlas_size = self.load_textures(ctx, texture_resolution, texture_list)
        texture_atlas.filter = mgl.NEAREST, mgl.NEAREST
        texture_atlas.use()

        # Set the "uniform" values the shaders require
        render_program["Mvp"].write(projection_matrix.astype('f4').tobytes())
        render_program["dir_light"].write(np.array(light_vector, dtype="f4").tobytes())
        render_program["atlas_size"].write(ctypes.c_uint32(atlas_size))
        render_program["tile_size"].write(ctypes.c_float(1 / atlas_size))

        return ctx, fbo, cube_vao

    ################################################################
    # Static helper methods
    ################################################################
    @staticmethod
    def _create_context():
        # TODO: EGL seems to need some commands first (currently manually executed)
        #  They probably must only be executed once per shell?
        #  apt install xvfb libgles2-mesa-dev
        #  export DISPLAY=:99.0
        #  Xvfb :99 -screen 0 640x480x24 &
        configurations = [
            {
                "standalone": True,
            },
            {
                "standalone": True,
                "backend": "egl",
            },
            {
                "standalone": True,
                "backend": "egl",
                "libgl": "libGL.so.1",
                "libegl": "libEGL.so.1",
            },
        ]
        exceptions = []
        for config in configurations:
            # noinspection PyBroadException
            try:
                ctx = mgl.create_context(**config)
            except Exception as e:
                exceptions.append(e)
                logger.debug("Could not create a moderngl context with %s" % str(config))
            else:
                logger.debug("Context created with %s" % str(config))
                return ctx

        logger.error("Could not create a modernGL context. Below is a list of the thrown exceptions")
        for e in exceptions:
            logger.error("\t%s: %s" % (str(type(e)), e))
        raise NoContextCreated(exceptions)

    @staticmethod
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
        with open(os.path.join(get_program_path(), path), "r") as fp:
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
                    raw_faces.append(tuple(x.split('/') for x in args))
                else:
                    pass

        # Combine all arrays and do the de-referencing
        data = np.array([
            value
            for face in raw_faces
            for vertex in face
            for value_list in [  # Each vertex consists of a position, normal and a uv
                raw_verticies[int(vertex[0]) - 1],
                raw_normals[int(vertex[2]) - 1],
                raw_uvs[int(vertex[1]) - 1],
            ]
            for value in value_list  # Combine all small arrays into a single large one
        ], dtype="f4")

        # This array is used to ensure faces always have the same id in later code.
        # This should only need to be changed if cube.obj changes
        # The index is the normal-index from cube.obj
        # The value is the index used later for this face
        face_mapping = [
            4,  # Top
            2,  # South
            3,  # West
            5,  # Bottom
            1,  # East
            0  # North
        ]

        print(raw_faces)
        face_indicies = np.array([
            face_mapping[int(vertex[2]) - 1]
            for face in raw_faces
            for vertex in face
        ], dtype="u4")

        # By reshaping the array all values can be read more easily
        # print(data.reshape((data.size // 8, 8)))
        # print(face_indicies)

        # Create a buffer containing the data
        cube_vbo = ctx.buffer(data.tobytes())
        cube_vbo_int = ctx.buffer(face_indicies.tobytes())
        # Create a VertexArray bound to the buffer and the render_program
        return ctx.vertex_array(
            render_program,
            [
                (cube_vbo, "3f4 3f4 2f4", "in_vert", "in_normal", "in_texcoord_0"),
                (cube_vbo_int, "u4", "in_faceid")
            ]
        )

    @staticmethod
    def load_textures(ctx, tile_size, texture_list: iter):
        # Unfortunatly we can't use a TextureArray. OpenGL 3 only supports the 3rd ize parameter to be
        # GL_MAX_ARRAY_TEXTURE_LAYERS long, which must be higher than 256 but we have over 600 Textures resulting
        # in more Textures than layers. On those systems (e.g. ubuntu in virtualbox all textures are black.
        # Workaround: Use a Texture-atlas instead
        # Calculate the (minimum) size of the atlas in sub-textures
        atlas_size = math.ceil(math.sqrt(len(texture_list)))
        gl_max_texture_size = ctx.info["GL_MAX_TEXTURE_SIZE"]
        # noinspection PyTypeChecker
        if type(gl_max_texture_size) not in (int, float):
            logger.warning("GL_MAX_TEXTURE_SIZE is no integer: %s" % gl_max_texture_size)
        elif tile_size * atlas_size > gl_max_texture_size:
            logger.error(
                "Can't create the TextureAtlas because it would be too large. "
                "Trying to do it anyway but this will probably result in a wrong Image. %s" % str({
                    "required_texture_size": tile_size * atlas_size,
                    "GL_MAX_TEXTURE_SIZE": ctx.info["GL_MAX_TEXTURE_SIZE"]
                })
            )
        texture_atlas = ctx.texture(
            size=(tile_size * atlas_size, tile_size * atlas_size),
            components=4,
            data=MGlRenderer.create_texture_atlas(tile_size, atlas_size, texture_list).tobytes()
        )
        return texture_atlas, atlas_size

    @staticmethod
    def create_texture_atlas(texture_size: int, atlas_size: int, texture_list: list):
        """
        This method combines a list of textures into a single image. All textures are resized to texture_size

        :param texture_size: Size of each individual texture
        :param atlas_size: Size of the resulting Image in subtexture cound
        :param texture_list: List of textures
        :return:
        """
        img = Image.new("RGBA", (texture_size*atlas_size, texture_size*atlas_size))
        for i, tex in enumerate(texture_list):
            x, y = (i % atlas_size) * texture_size, (i // atlas_size) * texture_size
            img.paste(tex, (x, y, x + texture_size, y + texture_size))
        return img

    # TODO: Move this code somewhere else to allow easier selection of pre-defined projection_matricies
    @staticmethod
    def calculate_projection_matrix():
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
    # Methods for rendering
    ################################################################
    def clear(self):
        self.ctx.clear()

    @contextmanager
    def colortint_pass(self):
        self.cube_model.program["is_colortint_pass"].write(ctypes.c_int32(1))
        yield
        self.cube_model.program["is_colortint_pass"].write(ctypes.c_int32(0))

    def render_cube(self, face_texture_ids: List[int], face_uvs: list, *,
                    pos=(0, 0, 0), model_rot=(0, 0, 0), scale=(1, 1, 1), uvlock=False,
                    rotation_origin=(0, 0, 0), rotation_axis=(1, 0, 0), rotation_angle=0,
                    face_rotation=(0, 0, 0, 0, 0, 0), face_tintindex=(0, 0, 0, 0, 0, 0)):
        """
        This method renders a cube. The result can be accessed using read_to_img

        :param face_texture_ids: This list contains the texture-IDs for each face (Order: N, E, S, W, Top, Bottom)
        :param face_uvs: This list contains the UV coordinates for each face. Each entry consists of 4 integers
                         (Order: N, E, S, W, Top, Bottom)
        :param pos: Position of the cube in local space
        :param model_rot: Rotation of the entire model (around (0, 0, 0))
        :param scale: Scale of the cube
        :param uvlock: Weather the UVs should be based on the world coordinates instead of face_uvs
        :param rotation_origin: Origin of the cube rotation
        :param rotation_axis: Axis around the cube is rotated
        :param rotation_angle: Angle defining how much the cube is rotated
        :param face_rotation: List of angles defining how much each face is rotated (Order: N, E, S, W, Top, Bottom)
        :param is_colortint_pass: Flag how the fragment shader should behave. True means only faces with tintindex are
                                  used. All other faces will be rendered black.
        :param face_tintindex: Which faces have a textureindex (Order: N, E, S, W, Top, Bottom)
        """
        # Write uniform values and render the vertex_array
        self.cube_model.program["face_texture_ids"].write(np.array(face_texture_ids, dtype="i4").tobytes())
        self.cube_model.program["face_uvs"].write(np.array(face_uvs, dtype="f4").tobytes())
        self.cube_model.program["face_rotation"].write(np.array(face_rotation, dtype="f4").tobytes())
        self.cube_model.program["model_rot"].write(np.array(model_rot, dtype="f4").tobytes())
        self.cube_model.program["uvlock"].write(ctypes.c_int32(1 if uvlock else 0))
        self.cube_model.program["pos"].write(np.array(pos, dtype="f4"))
        self.cube_model.program["scale"].write(np.array(scale, dtype="f4"))
        self.cube_model.program["rotation_angle"].write(ctypes.c_float(rotation_angle))
        self.cube_model.program["rotation_origin"].write(np.array(rotation_origin, dtype="f4").tobytes())
        self.cube_model.program["rotation_axis"].write(np.array(rotation_axis, dtype="f4").tobytes())
        self.cube_model.program["face_tintindex"].write(np.array(face_tintindex, dtype="i4").tobytes())
        self.cube_model.render()

    def read_to_img(self):
        # Read the data from the framebuffer and return it as a PIL.Image
        img = Image.frombytes("RGBA", (self.output_resolution, self.output_resolution), self.fbo.read(components=4))
        return img.transpose(Image.FLIP_TOP_BOTTOM)


class BlockRenderer(object):
    # Storage for finding the data value
    _data_map = defaultdict(list)

    def __init__(self, textures, *, block_list=None, start_block_id: int=1,
                 vertex_shader: str="overviewer_core/rendering/default.vert",
                 fragment_shader: str="overviewer_core/rendering/default.frag",
                 projection_matrix=None, output_resolution=24):
        # Not direclty related to rendering
        # TODO: Change the textures reference to a texturepath
        self.textures = textures
        self.assetLoader = AssetLoader(textures.find_file_local_path)
        self.start_block_id = start_block_id
        # TODO: Remove block_list
        if block_list is None:
            self.block_list = self.assetLoader.walk_assets(self.assetLoader.BLOCKSTATES_DIR, r".json")
        else:
            self.block_list = block_list

        # Configure rendering
        texture_list = self.get_all_textures()
        self.renderer = MGlRenderer(vertex_shader, fragment_shader, projection_matrix, texture_list,
                                    output_resolution=output_resolution)

    ################################################################
    # Finding and indexing textures
    ################################################################
    def get_all_textures(self):
        # TODO: Why does walk_assets sometimes return the file extension?
        # All textures have to be of size mc_texture_size*mc_texture_size
        paths = [
            name.rsplit('.', 1)[0]
            for name in self.assetLoader.walk_assets(self.assetLoader.TEXTURES_DIR + "/block", "")
        ]
        self.texture_indicies = {
            name: i
            for i, name in enumerate(paths)
        }
        return [
            self.assetLoader.load_img("block/" + name)
            for name in paths
        ]

        # return [
        #     self.assetLoader.load_img("block/stone"),
        #     self.assetLoader.load_img("block/melon_top"),
        #     self.assetLoader.load_img("block/oak_planks"),
        #     self.assetLoader.load_img("block/dirt"),
        #     self.assetLoader.load_img("block/melon_top"),
        #     self.assetLoader.load_img("block/bubble_coral_fan"),
        #     self.assetLoader.load_img("block/oak_sapling"),
        # ]

    def get_texture_index(self, name) -> int:
        # TODO: Implement this function
        # return 2
        return self.texture_indicies[name.rsplit('/', 1)[1]]

    ################################################################
    # Render methods
    ################################################################
    def render_element(self, element, texture_variables: dict, rotation_x_axis, rotation_y_axis, uvlock):
        _from = element["from"]
        _to = element["to"]
        # Calculate stupid default UVs if they are not given
        uv_default = {
            "north": (_to[0], 16-_to[1], _from[0], 16-_from[1]),
            "east": (_from[2], 16-_to[1], _to[2], 16-_from[1]),
            "south": (_from[0], 16-_to[1], _to[0], 16-_from[1]),
            "west": (_from[2], 16-_to[1], _to[2], 16-_from[1]),
            "up": (_from[0], _from[2], _to[0], _to[2]),
            "down": (_to[0], _from[2], _from[0], _to[2]),
        }

        # Convert the two cube corners into postion, and scale
        pos = tuple((t + f) / 32 - .5 for f, t in zip(element["from"], element["to"]))
        model_rot = (rotation_x_axis * pi / 180, rotation_y_axis * pi / 180)     # Not implemented yet
        scale = tuple((t - f) / 16 for f, t in zip(element["from"], element["to"]))
        face_texture_ids = [
            self.get_texture_index(texture_variables[element["faces"][face_name]["texture"][1:]])
            if face_name in element["faces"]
            else -1
            for face_name in ["north", "east", "south", "west", "up", "down"]
        ]
        face_uvs = [
            value / 16
            for face_name in ["north", "east", "south", "west", "up", "down"]
            for value in element["faces"].get(face_name, {}).get("uv", uv_default[face_name])
        ]
        face_tintindex = [
            element["faces"].get(face_name, {}).get("tintindex", -1)
            for face_name in ["north", "east", "south", "west", "up", "down"]
        ]
        face_rotation = [
            element["faces"][face_name].get("rotation", 0) * pi / 180
            if face_name in element["faces"]
            else 0
            for face_name in ["north", "east", "south", "west", "up", "down"]
        ]
        _rotation = element.get("rotation")
        axis_mapping = {
            'x': (1, 0, 0),
            'y': (0, 1, 0),
            'z': (0, 0, 1),
        }
        if _rotation is not None:
            rotation_kwargs = {
                "rotation_origin": tuple(x / 16 - 0.5 for x in _rotation["origin"]),
                "rotation_axis": axis_mapping[_rotation["axis"]],
                "rotation_angle": _rotation["angle"] * pi / 180,
            }
        else:
            rotation_kwargs = {}

        # Render the cube
        self.renderer.render_cube(
            face_texture_ids=face_texture_ids,
            face_uvs=face_uvs,
            pos=pos,
            model_rot=model_rot,
            scale=scale,
            uvlock=uvlock,
            face_rotation=face_rotation,
            face_tintindex=face_tintindex,
            **rotation_kwargs
        )

    def render_model(self, settings: dict) -> bool:
        model_data = self.assetLoader.load_and_combine_model(settings["model"])
        has_texturetint = False
        if "elements" not in model_data or model_data["elements"] is None:
            raise NoElementDataInDefinition
        for part in model_data["elements"]:
            self.render_element(
                element=part,
                texture_variables=model_data["textures"],
                rotation_x_axis=settings.get("x", 0),
                rotation_y_axis=settings.get("y", 0),
                uvlock=settings.get("uvlock", False),
            )

            print(part["faces"])
            has_texturetint = has_texturetint or any([
                "tintindex" in face
                for face in part["faces"].values()
            ])

        print(has_texturetint)
        return has_texturetint

    def _render_model_to_img(self, settings: dict):
        self.renderer.clear()
        has_texturetint = self.render_model(settings)
        return has_texturetint, self.renderer.read_to_img()

    def render_model_to_img(self, settings: dict) -> tuple:
        try:
            has_texturetint, img = self._render_model_to_img(settings)
            if has_texturetint:
                logger.debug("\tCreating colortint texture")
                with self.renderer.colortint_pass():
                    _, tint_texture = self._render_model_to_img(settings)
            else:
                tint_texture = None
        except NoElementDataInDefinition:
            logger.info("No Element data found for the model {0}".format(settings.get("model")))
            return None, None
        else:
            return img, tint_texture

    def render_blockstates(self, data: dict) -> (str, List[tuple]):
        if "variants" in data:
            for nbt_condition, variant in data["variants"].items():
                yield (nbt_condition, [
                    self.render_model_to_img(v) + (v.get("weight", 1), )
                    for v in (variant if type(variant) == list else (variant,))
                ])
        else:
            raise NotImplementedError("Multipart is not supported. Only blocks using 'variants' are.")

    ################################################################
    # NBT to Data conversion
    ################################################################
    def store_nbt_as_int(self, name, nbt_condition, blockid, data_value):
        compare_dict = {x.split("=")[0]: x.split("=")[1] for x in nbt_condition.split(',') if x != ""}
        self._data_map["minecraft:%s" % name].append((compare_dict, (blockid, data_value)))

    def get_data_map(self):
        return self._data_map

    @staticmethod
    def get_nbt_as_int(data_map: dict, key: str, properties: dict):
        entry = data_map.get(key)
        if entry is None:
            return None, None

        if properties is None:
            properties = {}

        for compare_dict, data_value in entry:
            for x in properties.items():
                if compare_dict.get(x[0], x[1]) != x[1]:
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
                        self.render_blockstates(self.assetLoader.load_blockstates(name=block_name))
                ):
                    yield block_index, block_name, nbt_index, nbt_condition, variants
            except NotImplementedError as e:
                if not ignore_unsupported_blocks:
                    raise e

    def iter_all_blocks(self, ignore_unsupported_blocks=True):

        logger.debug("Searching for blockstates in " + self.assetLoader.BLOCKSTATES_DIR)

        return self.iter_blocks(sorted(self.assetLoader.walk_assets(self.assetLoader.BLOCKSTATES_DIR, r".json")))

    def iter_for_generate(self):
        for block_index, block_name, nbt_index, nbt_condition, variants in self.iter_all_blocks():
            if len(variants) >= 1:
                logger.debug("Block found: {0} -> {1}:{2}".format(block_name, block_index, nbt_index))
                self.store_nbt_as_int(block_name, nbt_condition, block_index + self.start_block_id, nbt_index)
                # TODO: For now only return the first variant because multiple variants is currently not supported
                #  by overviewer.
                # TODO: For now only return the rendered Image and not the texturetint Image because the Textures
                #  class isn't expecting that.
                yield (block_index + self.start_block_id, nbt_index), variants[0][0]
