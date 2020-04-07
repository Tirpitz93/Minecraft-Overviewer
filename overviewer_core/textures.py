#    This file is part of the Minecraft Overviewer.
#
#    Minecraft Overviewer is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or (at
#    your option) any later version.
#
#    Minecraft Overviewer is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#    Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with the Overviewer.  If not, see <http://www.gnu.org/licenses/>.
from collections import OrderedDict
import sys
import imp
import os
import os.path
import zipfile
from io import BytesIO
import math
from random import randint
import numpy
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import logging
import functools

from . import util, texturegen
import logging

from .asset_loader import AssetLoader, AssetLoaderException

logger = logging.getLogger()

# global variables to collate information in @material decorators
blockmap_generators = {}

known_blocks = set()
used_datas = set()
max_blockid = 23000
max_data = 0

transparent_blocks = set()
solid_blocks = set([22000])
fluid_blocks = set()
nospawn_blocks = set()
nodata_blocks = set()

IMG_N =0

# This is here for circular import reasons.
# Please don't ask, I choose to repress these memories.
# ... okay fine I'll tell you.
# Initialising the C extension requires access to the globals above.
# Due to the circular import, this wouldn't work, unless we reload the
# module in the C extension or just move the import below its dependencies.
from .c_overviewer import alpha_over




color_map = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
             "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black"]


##
## Textures object
##
class Textures(object):
    """An object that generates a set of block sprites to use while
    rendering. It accepts a background color, north direction, and
    local textures path.
    """
    def __init__(self, texturepath=None, bgcolor=(26, 26, 26, 0), northdirection=0):
        self.bgcolor = bgcolor
        self.rotation = northdirection
        self.find_file_local_path = texturepath
        self.assetLoader = AssetLoader(texturepath)
        # not yet configurable
        self.texture_size = 24
        self.texture_dimensions = (self.texture_size, self.texture_size)
        
        # this is set in in generate()
        self.generated = False

        # see load_image_texture()
        # self.texture_cache = {}

        # once we find a jarfile that contains a texture, we cache the ZipFile object here

    
    ##
    ## pickle support
    ##

    def __getstate__(self):
        # we must get rid of the huge image lists, and other images
        attributes = self.__dict__.copy()
        for attr in ['assetLoader']:
            try:
                del attributes[attr]
            except KeyError:
                pass
        # attributes['assetLoader']['jars'] = OrderedDict()
        return attributes

    def __setstate__(self, attrs):
        # regenerate textures, if needed
        for attr, val in list(attrs.items()):
            setattr(self, attr, val)
        # self.texture_cache = {}
        self.assetLoader = AssetLoader(self.find_file_local_path)
        # if self.generated:
        #     self.generate()

    ##
    ## The big one: generate()
    ##
    
    def generate(self):
        # Make sure we have the foliage/grasscolor images available
        try:
            self.load_foliage_color()
            self.load_grass_color()
        except AssetLoaderException as e:
            logging.error(
                "Your system is missing either assets/minecraft/textures/colormap/foliage.png "
                "or assets/minecraft/textures/colormap/grass.png. Either complement your "
                "resource pack with these texture files, or install the vanilla Minecraft "
                "client to use as a fallback.")
            raise e
        
        # generate biome grass mask
        self.biome_grass_texture = self.build_block(self.assetLoader.load_image_texture(
        "assets/minecraft/textures/block/grass_block_top.png"), self.assetLoader.load_image_texture(
        "assets/minecraft/textures/block/grass_block_side_overlay.png"))
        
        # generate the blocks
        global blockmap_generators
        global known_blocks, used_datas
        global max_blockid, max_data

        # Get the maximum possible size when using automatic generation
        block_renderer = texturegen.BlockRenderer(self, start_block_id=22000)

        # Create Image Array
        self.blockmap = [None] * max_blockid * max_data

        for (blockid, data), texgen in list(blockmap_generators.items()):
            tex = texgen(self, blockid, data)
            self.blockmap[blockid * max_data + data] = self.generate_texture_tuple(tex)

        for (blockid, data), img in list(block_renderer.iter_for_generate()):
            self.blockmap[blockid * max_data + data] = self.generate_texture_tuple(img)
            known_blocks.add(blockid)



        if self.texture_size != 24:
            # rescale biome grass
            self.biome_grass_texture = self.biome_grass_texture.resize(self.texture_dimensions, Image.ANTIALIAS)
            
            # rescale the rest
            for i, tex in enumerate(self.blockmap):
                if tex is None:
                    continue
                block = tex[0]
                scaled_block = block.resize(self.texture_dimensions, Image.ANTIALIAS)
                self.blockmap[i] = self.generate_texture_tuple(scaled_block)

        self.generated = True
    
    ##
    ## Helpers for opening textures
    ##

    def load_water(self):
        """Special-case function for loading water."""
        watertexture = getattr(self, "watertexture", None)
        if watertexture:
            return watertexture
        watertexture = self.assetLoader.load_image_texture("assets/minecraft/textures/block/water_still.png")
        self.watertexture = watertexture
        return watertexture

    def load_lava(self):
        """Special-case function for loading lava."""
        lavatexture = getattr(self, "lavatexture", None)
        if lavatexture:
            return lavatexture
        lavatexture = self.assetLoader.load_image_texture("assets/minecraft/textures/block/lava_still.png")
        self.lavatexture = lavatexture
        return lavatexture
    
    def load_fire(self):
        """Special-case function for loading fire."""
        firetexture = getattr(self, "firetexture", None)
        if firetexture:
            return firetexture
        fireNS = self.assetLoader.load_image_texture("assets/minecraft/textures/block/fire_0.png")
        fireEW = self.assetLoader.load_image_texture("assets/minecraft/textures/block/fire_1.png")
        firetexture = (fireNS, fireEW)
        self.firetexture = firetexture
        return firetexture
    
    def load_portal(self):
        """Special-case function for loading portal."""
        portaltexture = getattr(self, "portaltexture", None)
        if portaltexture:
            return portaltexture
        portaltexture = self.assetLoader.load_image_texture("assets/minecraft/textures/block/nether_portal.png")
        self.portaltexture = portaltexture
        return portaltexture
    
    def load_light_color(self):
        """Helper function to load the light color texture."""
        if hasattr(self, "lightcolor"):
            return self.lightcolor
        try:
            lightcolor = list(self.assetLoader.load_image("light_normal.png").getdata())
        except Exception:
            logging.warning("Light color image could not be found.")
            lightcolor = None
        self.lightcolor = lightcolor
        return lightcolor
    
    def load_grass_color(self):
        """Helper function to load the grass color texture."""
        if not hasattr(self, "grasscolor"):
            self.grasscolor = list(self.assetLoader.load_image(
                "assets/minecraft/textures/colormap/grass.png").getdata())
        return self.grasscolor

    def load_foliage_color(self):
        """Helper function to load the foliage color texture."""
        if not hasattr(self, "foliagecolor"):
            self.foliagecolor = list(self.assetLoader.load_image(
                "assets/minecraft/textures/colormap/foliage.png").getdata())
        return self.foliagecolor

    #I guess "watercolor" is wrong. But I can't correct as my texture pack don't define water color.
    def load_water_color(self):
        """Helper function to load the water color texture."""
        if not hasattr(self, "watercolor"):
            self.watercolor = list(self.assetLoader.load_image("watercolor.png").getdata())
        return self.watercolor

    def _split_terrain(self, terrain):
        """Builds and returns a length 256 array of each 16x16 chunk
        of texture.
        """
        textures = []
        (terrain_width, terrain_height) = terrain.size
        texture_resolution = terrain_width / 16
        for y in range(16):
            for x in range(16):
                left = x*texture_resolution
                upper = y*texture_resolution
                right = left+texture_resolution
                lower = upper+texture_resolution
                region = terrain.transform(
                          (16, 16),
                          Image.EXTENT,
                          (left,upper,right,lower),
                          Image.BICUBIC)
                textures.append(region)

        return textures

    ##
    ## Image Transformation Functions
    ##

    @staticmethod
    def transform_image_top(img):
        """Takes a PIL image and rotates it left 45 degrees and shrinks the y axis
        by a factor of 2. Returns the resulting image, which will be 24x12 pixels

        """

        # Resize to 17x17, since the diagonal is approximately 24 pixels, a nice
        # even number that can be split in half twice
        img = img.resize((17, 17), Image.ANTIALIAS)

        # Build the Affine transformation matrix for this perspective
        transform = numpy.matrix(numpy.identity(3))
        # Translate up and left, since rotations are about the origin
        transform *= numpy.matrix([[1,0,8.5],[0,1,8.5],[0,0,1]])
        # Rotate 45 degrees
        ratio = math.cos(math.pi/4)
        #transform *= numpy.matrix("[0.707,-0.707,0;0.707,0.707,0;0,0,1]")
        transform *= numpy.matrix([[ratio,-ratio,0],[ratio,ratio,0],[0,0,1]])
        # Translate back down and right
        transform *= numpy.matrix([[1,0,-12],[0,1,-12],[0,0,1]])
        # scale the image down by a factor of 2
        transform *= numpy.matrix("[1,0,0;0,2,0;0,0,1]")

        transform = numpy.array(transform)[:2,:].ravel().tolist()

        newimg = img.transform((24,12), Image.AFFINE, transform)
        return newimg

    @staticmethod
    def transform_image_side(img):
        """Takes an image and shears it for the left side of the cube (reflect for
        the right side)"""

        # Size of the cube side before shear
        img = img.resize((12,12), Image.ANTIALIAS)

        # Apply shear
        transform = numpy.matrix(numpy.identity(3))
        transform *= numpy.matrix("[1,0,0;-0.5,1,0;0,0,1]")

        transform = numpy.array(transform)[:2,:].ravel().tolist()

        newimg = img.transform((12,18), Image.AFFINE, transform)
        return newimg

    @staticmethod
    def transform_image_slope(img):
        """Takes an image and shears it in the shape of a slope going up
        in the -y direction (reflect for +x direction). Used for minetracks"""

        # Take the same size as trasform_image_side
        img = img.resize((12,12), Image.ANTIALIAS)

        # Apply shear
        transform = numpy.matrix(numpy.identity(3))
        transform *= numpy.matrix("[0.75,-0.5,3;0.25,0.5,-3;0,0,1]")
        transform = numpy.array(transform)[:2,:].ravel().tolist()

        newimg = img.transform((24,24), Image.AFFINE, transform)

        return newimg


    @staticmethod
    def transform_image_angle(img, angle):
        """Takes an image an shears it in arbitrary angle with the axis of
        rotation being vertical.

        WARNING! Don't use angle = pi/2 (or multiplies), it will return
        a blank image (or maybe garbage).

        NOTE: angle is in the image not in game, so for the left side of a
        block angle = 30 degree.
        """

        # Take the same size as trasform_image_side
        img = img.resize((12,12), Image.ANTIALIAS)

        # some values
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        # function_x and function_y are used to keep the result image in the 
        # same position, and constant_x and constant_y are the coordinates
        # for the center for angle = 0.
        constant_x = 6.
        constant_y = 6.
        function_x = 6.*(1-cos_angle)
        function_y = -6*sin_angle
        big_term = ( (sin_angle * (function_x + constant_x)) - cos_angle* (function_y + constant_y))/cos_angle

        # The numpy array is not really used, but is helpful to 
        # see the matrix used for the transformation.
        transform = numpy.array([[1./cos_angle, 0, -(function_x + constant_x)/cos_angle],
                                 [-sin_angle/(cos_angle), 1., big_term ],
                                 [0, 0, 1.]])

        transform = tuple(transform[0]) + tuple(transform[1])

        newimg = img.transform((24,24), Image.AFFINE, transform)

        return newimg


    def build_block(self, top, side):
        """From a top texture and a side texture, build a block image.
        top and side should be 16x16 image objects. Returns a 24x24 image

        """
        img = Image.new("RGBA", (24,24), self.bgcolor)

        original_texture = top.copy()
        top = self.transform_image_top(top)

        if not side:
            alpha_over(img, top, (0,0), top)
            return img

        side = self.transform_image_side(side)
        otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

        # Darken the sides slightly. These methods also affect the alpha layer,
        # so save them first (we don't want to "darken" the alpha layer making
        # the block transparent)
        sidealpha = side.split()[3]
        side = ImageEnhance.Brightness(side).enhance(0.9)
        side.putalpha(sidealpha)
        othersidealpha = otherside.split()[3]
        otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
        otherside.putalpha(othersidealpha)

        alpha_over(img, top, (0,0), top)
        alpha_over(img, side, (0,6), side)
        alpha_over(img, otherside, (12,6), otherside)

        # Manually touch up 6 pixels that leave a gap because of how the
        # shearing works out. This makes the blocks perfectly tessellate-able
        for x,y in [(13,23), (17,21), (21,19)]:
            # Copy a pixel to x,y from x-1,y
            img.putpixel((x,y), img.getpixel((x-1,y)))
        for x,y in [(3,4), (7,2), (11,0)]:
            # Copy a pixel to x,y from x+1,y
            img.putpixel((x,y), img.getpixel((x+1,y)))

        return img

    def build_slab_block(self, top, side, upper):
        """From a top texture and a side texture, build a slab block image.
        top and side should be 16x16 image objects. Returns a 24x24 image

        """
        # cut the side texture in half
        mask = side.crop((0,8,16,16))
        side = Image.new(side.mode, side.size, self.bgcolor)
        alpha_over(side, mask,(0,0,16,8), mask)

        # plain slab
        top = self.transform_image_top(top)
        side = self.transform_image_side(side)
        otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

        sidealpha = side.split()[3]
        side = ImageEnhance.Brightness(side).enhance(0.9)
        side.putalpha(sidealpha)
        othersidealpha = otherside.split()[3]
        otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
        otherside.putalpha(othersidealpha)

        # upside down slab
        delta = 0
        if upper:
            delta = 6

        img = Image.new("RGBA", (24,24), self.bgcolor)
        alpha_over(img, side, (0,12 - delta), side)
        alpha_over(img, otherside, (12,12 - delta), otherside)
        alpha_over(img, top, (0,6 - delta), top)

        # Manually touch up 6 pixels that leave a gap because of how the
        # shearing works out. This makes the blocks perfectly tessellate-able
        if upper:
            for x,y in [(3,4), (7,2), (11,0)]:
                # Copy a pixel to x,y from x+1,y
                img.putpixel((x,y), img.getpixel((x+1,y)))
            for x,y in [(13,17), (17,15), (21,13)]:
                # Copy a pixel to x,y from x-1,y
                img.putpixel((x,y), img.getpixel((x-1,y)))
        else:
            for x,y in [(3,10), (7,8), (11,6)]:
                # Copy a pixel to x,y from x+1,y
                img.putpixel((x,y), img.getpixel((x+1,y)))
            for x,y in [(13,23), (17,21), (21,19)]:
                # Copy a pixel to x,y from x-1,y
                img.putpixel((x,y), img.getpixel((x-1,y)))

        return img

    def build_full_block(self, top, side1, side2, side3, side4, bottom=None):
        """From a top texture, a bottom texture and 4 different side textures,
        build a full block with four differnts faces. All images should be 16x16 
        image objects. Returns a 24x24 image. Can be used to render any block.

        side1 is in the -y face of the cube     (top left, east)
        side2 is in the +x                      (top right, south)
        side3 is in the -x                      (bottom left, north)
        side4 is in the +y                      (bottom right, west)

        A non transparent block uses top, side 3 and side 4.

        If top is a tuple then first item is the top image and the second
        item is an increment (integer) from 0 to 16 (pixels in the
        original minecraft texture). This increment will be used to crop the
        side images and to paste the top image increment pixels lower, so if
        you use an increment of 8, it will draw a half-block.

        NOTE: this method uses the bottom of the texture image (as done in 
        minecraft with beds and cackes)

        """

        increment = 0
        if isinstance(top, tuple):
            increment = int(round((top[1] / 16.)*12.)) # range increment in the block height in pixels (half texture size)
            crop_height = increment
            top = top[0]
            if side1 is not None:
                side1 = side1.copy()
                ImageDraw.Draw(side1).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))
            if side2 is not None:
                side2 = side2.copy()
                ImageDraw.Draw(side2).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))
            if side3 is not None:
                side3 = side3.copy()
                ImageDraw.Draw(side3).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))
            if side4 is not None:
                side4 = side4.copy()
                ImageDraw.Draw(side4).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))

        img = Image.new("RGBA", (24,24), self.bgcolor)

        # first back sides
        if side1 is not None :
            side1 = self.transform_image_side(side1)
            side1 = side1.transpose(Image.FLIP_LEFT_RIGHT)

            # Darken this side.
            sidealpha = side1.split()[3]
            side1 = ImageEnhance.Brightness(side1).enhance(0.9)
            side1.putalpha(sidealpha)        

            alpha_over(img, side1, (0,0), side1)


        if side2 is not None :
            side2 = self.transform_image_side(side2)

            # Darken this side.
            sidealpha2 = side2.split()[3]
            side2 = ImageEnhance.Brightness(side2).enhance(0.8)
            side2.putalpha(sidealpha2)

            alpha_over(img, side2, (12,0), side2)

        if bottom is not None :
            bottom = self.transform_image_top(bottom)
            alpha_over(img, bottom, (0,12), bottom)

        # front sides
        if side3 is not None :
            side3 = self.transform_image_side(side3)

            # Darken this side
            sidealpha = side3.split()[3]
            side3 = ImageEnhance.Brightness(side3).enhance(0.9)
            side3.putalpha(sidealpha)

            alpha_over(img, side3, (0,6), side3)

        if side4 is not None :
            side4 = self.transform_image_side(side4)
            side4 = side4.transpose(Image.FLIP_LEFT_RIGHT)

            # Darken this side
            sidealpha = side4.split()[3]
            side4 = ImageEnhance.Brightness(side4).enhance(0.8)
            side4.putalpha(sidealpha)

            alpha_over(img, side4, (12,6), side4)

        if top is not None :
            top = self.transform_image_top(top)
            alpha_over(img, top, (0, increment), top)

        # Manually touch up 6 pixels that leave a gap because of how the
        # shearing works out. This makes the blocks perfectly tessellate-able
        for x,y in [(13,23), (17,21), (21,19)]:
            # Copy a pixel to x,y from x-1,y
            img.putpixel((x,y), img.getpixel((x-1,y)))
        for x,y in [(3,4), (7,2), (11,0)]:
            # Copy a pixel to x,y from x+1,y
            img.putpixel((x,y), img.getpixel((x+1,y)))
        global IMG_N
        # IMG_N +=1
        # img.save("C:/Datafile/LSelter/Documents/Minecraft-Overviewer/test_conf/debug/"+ str(IMG_N) + ".png")
        return img

    def build_sprite(self, side):
        """From a side texture, create a sprite-like texture such as those used
        for spiderwebs or flowers."""
        img = Image.new("RGBA", (24,24), self.bgcolor)

        side = self.transform_image_side(side)
        otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

        alpha_over(img, side, (6,3), side)
        alpha_over(img, otherside, (6,3), otherside)
        return img

    def build_billboard(self, tex):
        """From a texture, create a billboard-like texture such as
        those used for tall grass or melon stems.
        """
        img = Image.new("RGBA", (24,24), self.bgcolor)

        front = tex.resize((14, 12), Image.ANTIALIAS)
        alpha_over(img, front, (5,9))
        return img

    def generate_opaque_mask(self, img):
        """ Takes the alpha channel of the image and generates a mask
        (used for lighting the block) that deprecates values of alpha
        smallers than 50, and sets every other value to 255. """

        alpha = img.split()[3]
        return alpha.point(lambda a: int(min(a, 25.5) * 10))

    def tint_texture(self, im, c):
        # apparently converting to grayscale drops the alpha channel?
        i = ImageOps.colorize(ImageOps.grayscale(im), (0,0,0), c)
        i.putalpha(im.split()[3]); # copy the alpha band back in. assuming RGBA
        return i

    def generate_texture_tuple(self, img):
        """ This takes an image and returns the needed tuple for the
        blockmap array."""
        if img is None:
            return None
        return (img, self.generate_opaque_mask(img))

##
## The other big one: @material and associated framework
##

# the material registration decorator
def material(blockid=[], data=[0], **kwargs):
    # mapping from property name to the set to store them in
    properties = {"transparent" : transparent_blocks, "solid" : solid_blocks, "fluid" : fluid_blocks, "nospawn" : nospawn_blocks, "nodata" : nodata_blocks}
    
    # make sure blockid and data are iterable
    try:
        iter(blockid)
    except Exception:
        blockid = [blockid,]
    try:
        iter(data)
    except Exception:
        data = [data,]
        
    def inner_material(func):
        global blockmap_generators
        global max_data, max_blockid

        # create a wrapper function with a known signature
        @functools.wraps(func)
        def func_wrapper(texobj, blockid, data):
            return func(texobj, blockid, data)
        
        used_datas.update(data)
        if max(data) >= max_data:
            max_data = max(data) + 1
        
        for block in blockid:
            # set the property sets appropriately
            known_blocks.update([block])
            if block >= max_blockid:
                max_blockid = block + 1
            for prop in properties:
                try:
                    if block in kwargs.get(prop, []):
                        properties[prop].update([block])
                except TypeError:
                    if kwargs.get(prop, False):
                        properties[prop].update([block])
            
            # populate blockmap_generators with our function
            for d in data:
                blockmap_generators[(block, d)] = func_wrapper
        
        return func_wrapper
    return inner_material

# shortcut function for pure blocks, default to solid, nodata
def block(blockid=[], top_image=None, side_image=None, **kwargs):
    new_kwargs = {'solid' : True, 'nodata' : True}
    new_kwargs.update(kwargs)
    
    if top_image is None:
        raise ValueError("top_image was not provided")
    
    if side_image is None:
        side_image = top_image
    
    @material(blockid=blockid, **new_kwargs)
    def inner_block(self, unused_id, unused_data):
        return self.build_block(self.assetLoader.load_image_texture(top_image), self.assetLoader.load_image_texture(side_image))
    return inner_block

# shortcut function for sprite blocks, defaults to transparent, nodata
def sprite(blockid=[], imagename=None, **kwargs):
    new_kwargs = {'transparent' : True, 'nodata' : True}
    new_kwargs.update(kwargs)
    
    if imagename is None:
        raise ValueError("imagename was not provided")
    
    @material(blockid=blockid, **new_kwargs)
    def inner_sprite(self, unused_id, unused_data):
        return self.build_sprite(self.assetLoader.load_image_texture(imagename))
    return inner_sprite

# shortcut function for billboard blocks, defaults to transparent, nodata
def billboard(blockid=[], imagename=None, **kwargs):
    new_kwargs = {'transparent' : True, 'nodata' : True}
    new_kwargs.update(kwargs)
    
    if imagename is None:
        raise ValueError("imagename was not provided")
    
    @material(blockid=blockid, **new_kwargs)
    def inner_billboard(self, unused_id, unused_data):
        return self.build_billboard(self.assetLoader.load_image_texture(imagename))
    return inner_billboard


##
## and finally: actual texture definitions
##

# water, glass, and ice (no inner surfaces)
# uses pseudo-ancildata found in iterate.c
@material(blockid=[8, 9, 20, 79, 95], data=list(range(512)), fluid=(8, 9), transparent=True, nospawn=True, solid=(79, 20, 95))
def no_inner_surfaces(self, blockid, data):
    if blockid == 8 or blockid == 9:
        texture = self.load_water()
    # elif blockid == 20:
    #     texture = self.assetLoader.load_image_texture("assets/minecraft/textures/block/glass.png")
    elif blockid == 95:
        texture = self.assetLoader.load_image_texture("assets/minecraft/textures/block/%s_stained_glass.png" % color_map[data & 0x0f])
    else:
        texture = self.assetLoader.load_image_texture("assets/minecraft/textures/block/ice.png")

    # now that we've used the lower 4 bits to get color, shift down to get the 5 bits that encode face hiding
    if not (blockid == 8 or blockid == 9): # water doesn't have a shifted pseudodata
        data = data >> 4

    if (data & 0b10000) == 16:
        top = texture
    else:
        top = None
        
    if (data & 0b0001) == 1:
        side1 = texture    # top left
    else:
        side1 = None
    
    if (data & 0b1000) == 8:
        side2 = texture    # top right           
    else:
        side2 = None
    
    if (data & 0b0010) == 2:
        side3 = texture    # bottom left    
    else:
        side3 = None
    
    if (data & 0b0100) == 4:
        side4 = texture    # bottom right
    else:
        side4 = None
    
    # if nothing shown do not draw at all
    if top is None and side3 is None and side4 is None:
        return None
    
    img = self.build_full_block(top,None,None,side3,side4)
    return img

@material(blockid=[10, 11], data=list(range(16)), fluid=True, transparent=False, nospawn=True)
def lava(self, blockid, data):
    lavatex = self.load_lava()
    return self.build_block(lavatex, lavatex)

@material(blockid=26, data=list(range(12)), transparent=True, nospawn=True)
def bed(self, blockid, data):
    # first get rotation done
    # Masked to not clobber block head/foot info
    if self.rotation == 1:
        if (data & 0b0011) == 0: data = data & 0b1100 | 1
        elif (data & 0b0011) == 1: data = data & 0b1100 | 2
        elif (data & 0b0011) == 2: data = data & 0b1100 | 3
        elif (data & 0b0011) == 3: data = data & 0b1100 | 0
    elif self.rotation == 2:
        if (data & 0b0011) == 0: data = data & 0b1100 | 2
        elif (data & 0b0011) == 1: data = data & 0b1100 | 3
        elif (data & 0b0011) == 2: data = data & 0b1100 | 0
        elif (data & 0b0011) == 3: data = data & 0b1100 | 1
    elif self.rotation == 3:
        if (data & 0b0011) == 0: data = data & 0b1100 | 3
        elif (data & 0b0011) == 1: data = data & 0b1100 | 0
        elif (data & 0b0011) == 2: data = data & 0b1100 | 1
        elif (data & 0b0011) == 3: data = data & 0b1100 | 2
    
    bed_texture = self.assetLoader.load_image("assets/minecraft/textures/entity/bed/red.png") # FIXME: do tile entity colours
    increment = 8
    left_face = None
    right_face = None
    top_face = None
    if data & 0x8 == 0x8: # head of the bed
        top = bed_texture.copy().crop((6,6,22,22))

        # Composing the side
        side = Image.new("RGBA", (16,16))
        side_part1 = bed_texture.copy().crop((0,6,6,22)).rotate(90, expand=True)
        # foot of the bed
        side_part2 = bed_texture.copy().crop((53,3,56,6))
        side_part2_f = side_part2.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(side, side_part1, (0,7), side_part1)
        alpha_over(side, side_part2, (0,13), side_part2)

        end = Image.new("RGBA", (16,16))
        end_part = bed_texture.copy().crop((6,0,22,6)).rotate(180)
        alpha_over(end, end_part, (0,7), end_part)
        alpha_over(end, side_part2, (0,13), side_part2)
        alpha_over(end, side_part2_f, (13,13), side_part2_f)
        if data & 0x00 == 0x00: # South
            top_face = top.rotate(180)
            left_face = side.transpose(Image.FLIP_LEFT_RIGHT)
            right_face = end
        if data & 0x01 == 0x01: # West
            top_face = top.rotate(90)
            left_face = end
            right_face = side.transpose(Image.FLIP_LEFT_RIGHT)
        if data & 0x02 == 0x02: # North
            top_face = top
            left_face = side
        if data & 0x03 == 0x03: # East
            top_face = top.rotate(270)
            right_face = side
    
    else: # foot of the bed
        top = bed_texture.copy().crop((6,28,22,44))
        side = Image.new("RGBA", (16,16))
        side_part1 = bed_texture.copy().crop((0,28,6,44)).rotate(90, expand=True)
        side_part2 = bed_texture.copy().crop((53,3,56,6))
        side_part2_f = side_part2.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(side, side_part1, (0,7), side_part1)
        alpha_over(side, side_part2, (13,13), side_part2)

        end = Image.new("RGBA", (16,16))
        end_part = bed_texture.copy().crop((22,22,38,28)).rotate(180)
        alpha_over(end, end_part, (0,7), end_part)
        alpha_over(end, side_part2, (0,13), side_part2)
        alpha_over(end, side_part2_f, (13,13), side_part2_f)
        if data & 0x00 == 0x00: # South
            top_face = top.rotate(180)
            left_face = side.transpose(Image.FLIP_LEFT_RIGHT)
        if data & 0x01 == 0x01: # West
            top_face = top.rotate(90)
            right_face = side.transpose(Image.FLIP_LEFT_RIGHT)
        if data & 0x02 == 0x02: # North
            top_face = top
            left_face = side
            right_face = end
        if data & 0x03 == 0x03: # East
            top_face = top.rotate(270)
            left_face = end
            right_face = side

    top_face = (top_face, increment)
    return self.build_full_block(top_face, None, None, left_face, right_face)

# flowers
@material(blockid=38, data=list(range(10)), transparent=True)
def flower(self, blockid, data):
    flower_map = ["poppy", "blue_orchid", "allium", "azure_bluet", "red_tulip", "orange_tulip",
                  "white_tulip", "pink_tulip", "oxeye_daisy", "dandelion"]
    texture = self.assetLoader.load_image_texture("assets/minecraft/textures/block/%s.png" % flower_map[data])

    return self.build_billboard(texture)

# bamboo
@material(blockid=11416, transparent=True)
def bamboo(self, blockid, data):
    # get the  multipart texture of the lantern
    inputtexture = self.assetLoader.load_image_texture("assets/minecraft/textures/block/bamboo_stalk.png")

    # # now create a textures, using the parts defined in bamboo1_age0.json
        # {   "from": [ 7, 0, 7 ],
        #     "to": [ 9, 16, 9 ],
        #     "faces": {
        #         "down":  { "uv": [ 13, 4, 15, 6 ], "texture": "#all", "cullface": "down" },
        #         "up":    { "uv": [ 13, 0, 15, 2], "texture": "#all", "cullface": "up" },
        #         "north": { "uv": [ 0, 0, 2, 16 ], "texture": "#all" },
        #         "south": { "uv": [ 0, 0, 2, 16 ], "texture": "#all" },
        #         "west":  { "uv": [  0, 0, 2, 16 ], "texture": "#all" },
        #         "east":  { "uv": [  0, 0, 2, 16 ], "texture": "#all" }
        #     }
        # }

    side_crop = inputtexture.crop((0, 0, 3, 16))
    side_slice = side_crop.copy()
    side_texture = Image.new("RGBA", (16, 16), self.bgcolor)
    side_texture.paste(side_slice,(0, 0))

    # JSON data for top
    # "up":    { "uv": [ 13, 0, 15, 2], "texture": "#all", "cullface": "up" },
    top_crop = inputtexture.crop((13, 0, 16, 3))
    top_slice = top_crop.copy()
    top_texture = Image.new("RGBA", (16, 16), self.bgcolor)
    top_texture.paste(top_slice,(5, 5))

    # mimic parts of build_full_block, to get an object smaller than a block 
    # build_full_block(self, top, side1, side2, side3, side4, bottom=None):
    # a non transparent block uses top, side 3 and side 4.
    img = Image.new("RGBA", (24, 24), self.bgcolor)
    # prepare the side textures
    # side3
    side3 = self.transform_image_side(side_texture)
    # Darken this side
    sidealpha = side3.split()[3]
    side3 = ImageEnhance.Brightness(side3).enhance(0.9)
    side3.putalpha(sidealpha)
    # place the transformed texture
    xoff = 3
    yoff = 0
    alpha_over(img, side3, (4+xoff, yoff), side3)
    # side4
    side4 = self.transform_image_side(side_texture)
    side4 = side4.transpose(Image.FLIP_LEFT_RIGHT)
    # Darken this side
    sidealpha = side4.split()[3]
    side4 = ImageEnhance.Brightness(side4).enhance(0.8)
    side4.putalpha(sidealpha)
    alpha_over(img, side4, (-4+xoff, yoff), side4)
    # top
    top = self.transform_image_top(top_texture)
    alpha_over(img, top, (-4+xoff, -5), top)
    return img

# composter
@material(blockid=11417, data=list(range(9)), transparent=True)
def composter(self, blockid, data):
    side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/composter_side.png")
    top = self.assetLoader.load_image_texture("assets/minecraft/textures/block/composter_top.png")
    # bottom = self.assetLoader.load_image_texture("assets/minecraft/textures/block/composter_bottom.png")

    if data == 0:  # empty
        return self.build_full_block(top, side, side, side, side)

    if data == 8:
        compost = self.transform_image_top(
            self.assetLoader.load_image_texture("assets/minecraft/textures/block/composter_ready.png"))
    else:
        compost = self.transform_image_top(
            self.assetLoader.load_image_texture("assets/minecraft/textures/block/composter_compost.png"))

    nudge = {1: (0, 9), 2: (0, 8), 3: (0, 7), 4: (0, 6), 5: (0, 4), 6: (0, 2), 7: (0, 0), 8: (0, 0)}

    img = self.build_full_block(None, side, side, None, None)
    alpha_over(img, compost, nudge[data], compost)
    img2 = self.build_full_block(top, None, None, side, side)
    alpha_over(img, img2, (0, 0), img2)
    return img

# fire
@material(blockid=51, data=list(range(16)), transparent=True)
def fire(self, blockid, data):
    firetextures = self.load_fire()
    side1 = self.transform_image_side(firetextures[0])
    side2 = self.transform_image_side(firetextures[1]).transpose(Image.FLIP_LEFT_RIGHT)
    
    img = Image.new("RGBA", (24,24), self.bgcolor)

    alpha_over(img, side1, (12,0), side1)
    alpha_over(img, side2, (0,0), side2)

    alpha_over(img, side1, (0,6), side1)
    alpha_over(img, side2, (12,6), side2)
    
    return img

# normal, locked (used in april's fool day), ender and trapped chest
# NOTE:  locked chest used to be id95 (which is now stained glass)
@material(blockid=[54, 130, 146], data=list(range(30)), transparent = True)
def chests(self, blockid, data):
    # the first 3 bits are the orientation as stored in minecraft, 
    # bits 0x8 and 0x10 indicate which half of the double chest is it.

    # first, do the rotation if needed
    orientation_data = data & 7
    if self.rotation == 1:
        if orientation_data == 2: data = 5 | (data & 24)
        elif orientation_data == 3: data = 4 | (data & 24)
        elif orientation_data == 4: data = 2 | (data & 24)
        elif orientation_data == 5: data = 3 | (data & 24)
    elif self.rotation == 2:
        if orientation_data == 2: data = 3 | (data & 24)
        elif orientation_data == 3: data = 2 | (data & 24)
        elif orientation_data == 4: data = 5 | (data & 24)
        elif orientation_data == 5: data = 4 | (data & 24)
    elif self.rotation == 3:
        if orientation_data == 2: data = 4 | (data & 24)
        elif orientation_data == 3: data = 5 | (data & 24)
        elif orientation_data == 4: data = 3 | (data & 24)
        elif orientation_data == 5: data = 2 | (data & 24)
    
    if blockid == 130 and not data in [2, 3, 4, 5]: return None
        # iterate.c will only return the ancil data (without pseudo 
        # ancil data) for locked and ender chests, so only 
        # ancilData = 2,3,4,5 are used for this blockids
    
    if data & 24 == 0:
        if blockid == 130: t = self.assetLoader.load_image("assets/minecraft/textures/entity/chest/ender.png")
        else:
            try:
                t = self.assetLoader.load_image("assets/minecraft/textures/entity/chest/normal.png")
            except (AssetLoaderException, IOError):
                t = self.assetLoader.load_image("assets/minecraft/textures/entity/chest/chest.png")

        t = ImageOps.flip(t) # for some reason the 1.15 images are upside down

        # the textures is no longer in terrain.png, get it from
        # item/chest.png and get by cropping all the needed stuff
        if t.size != (64, 64): t = t.resize((64, 64), Image.ANTIALIAS)
        # top
        top = t.crop((28, 50, 42, 64))
        top.load() # every crop need a load, crop is a lazy operation
                   # see PIL manual
        img = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(img, top, (1, 1))
        top = img
        # front
        front_top = t.crop((42, 45, 56, 50))
        front_top.load()
        front_bottom = t.crop((42, 21, 56, 31))
        front_bottom.load()
        front_lock = t.crop((1, 59, 3, 63))
        front_lock.load()
        front = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(front, front_top, (1, 1))
        alpha_over(front, front_bottom, (1, 5))
        alpha_over(front, front_lock, (7, 3))
        # left side
        # left side, right side, and back are essentially the same for
        # the default texture, we take it anyway just in case other
        # textures make use of it.
        side_l_top = t.crop((14, 45, 28, 50))
        side_l_top.load()
        side_l_bottom = t.crop((14, 21, 28, 31))
        side_l_bottom.load()
        side_l = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(side_l, side_l_top, (1, 1))
        alpha_over(side_l, side_l_bottom, (1, 5))
        # right side
        side_r_top = t.crop((28, 45, 42, 50))
        side_r_top.load()
        side_r_bottom = t.crop((28, 21, 42, 31))
        side_r_bottom.load()
        side_r = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(side_r, side_r_top, (1, 1))
        alpha_over(side_r, side_r_bottom, (1, 5))
        # back
        back_top = t.crop((0, 45, 14, 50))
        back_top.load()
        back_bottom = t.crop((0, 21, 14, 31))
        back_bottom.load()
        back = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(back, back_top, (1, 1))
        alpha_over(back, back_bottom, (1, 5))

    else:
        # large chest
        # the textures is no longer in terrain.png, get it from 
        # item/chest.png and get all the needed stuff
        t_left = self.assetLoader.load_image("assets/minecraft/textures/entity/chest/normal_left.png")
        t_right = self.assetLoader.load_image("assets/minecraft/textures/entity/chest/normal_right.png")
        # for some reason the 1.15 images are upside down
        t_left = ImageOps.flip(t_left)
        t_right = ImageOps.flip(t_right)

        # Top
        top_left = t_right.crop((29, 50, 44, 64))
        top_left.load()
        top_right = t_left.crop((29, 50, 44, 64))
        top_right.load()

        top = Image.new("RGBA", (32, 16), self.bgcolor)
        alpha_over(top,top_left, (1, 1))
        alpha_over(top,top_right, (16, 1))

        # Front
        front_top_left = t_left.crop((43, 45, 58, 50))
        front_top_left.load()
        front_top_right = t_right.crop((43, 45, 58, 50))
        front_top_right.load()

        front_bottom_left = t_left.crop((43, 21, 58, 31))
        front_bottom_left.load()
        front_bottom_right = t_right.crop((43, 21, 58, 31))
        front_bottom_right.load()

        front_lock = t_left.crop((1, 59, 3, 63))
        front_lock.load()

        front = Image.new("RGBA", (32, 16), self.bgcolor)
        alpha_over(front, front_top_left, (1, 1))
        alpha_over(front, front_top_right, (16, 1))
        alpha_over(front, front_bottom_left, (1, 5))
        alpha_over(front, front_bottom_right, (16, 5))
        alpha_over(front, front_lock, (15, 3))

        # Back
        back_top_left = t_right.crop((14, 45, 29, 50))
        back_top_left.load()
        back_top_right = t_left.crop((14, 45, 29, 50))
        back_top_right.load()

        back_bottom_left = t_right.crop((14, 21, 29, 31))
        back_bottom_left.load()
        back_bottom_right = t_left.crop((14, 21, 29, 31))
        back_bottom_right.load()

        back = Image.new("RGBA", (32, 16), self.bgcolor)
        alpha_over(back, back_top_left, (1, 1))
        alpha_over(back, back_top_right, (16, 1))
        alpha_over(back, back_bottom_left, (1, 5))
        alpha_over(back, back_bottom_right, (16, 5))
        
        # left side
        side_l_top = t_left.crop((29, 45, 43, 50))
        side_l_top.load()
        side_l_bottom = t_left.crop((29, 21, 43, 31))
        side_l_bottom.load()
        side_l = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(side_l, side_l_top, (1, 1))
        alpha_over(side_l, side_l_bottom, (1, 5))
        # right side
        side_r_top = t_right.crop((0, 45, 14, 50))
        side_r_top.load()
        side_r_bottom = t_right.crop((0, 21, 14, 31))
        side_r_bottom.load()
        side_r = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(side_r, side_r_top, (1, 1))
        alpha_over(side_r, side_r_bottom, (1, 5))

        if data & 24 == 8: # double chest, first half
            top = top.crop((0, 0, 16, 16))
            top.load()
            front = front.crop((0, 0, 16, 16))
            front.load()
            back = back.crop((0, 0, 16, 16))
            back.load()
            #~ side = side_l

        elif data & 24 == 16: # double, second half
            top = top.crop((16, 0, 32, 16))
            top.load()
            front = front.crop((16, 0, 32, 16))
            front.load()
            back = back.crop((16, 0, 32, 16))
            back.load()
            #~ side = side_r

        else: # just in case
            return None

    # compose the final block
    img = Image.new("RGBA", (24, 24), self.bgcolor)
    if data & 7 == 2: # north
        side = self.transform_image_side(side_r)
        alpha_over(img, side, (1, 7))
        back = self.transform_image_side(back)
        alpha_over(img, back.transpose(Image.FLIP_LEFT_RIGHT), (11, 7))
        front = self.transform_image_side(front)
        top = self.transform_image_top(top.rotate(180))
        alpha_over(img, top, (0, 2))

    elif data & 7 == 3: # south
        side = self.transform_image_side(side_l)
        alpha_over(img, side, (1, 7))
        front = self.transform_image_side(front).transpose(Image.FLIP_LEFT_RIGHT)
        top = self.transform_image_top(top.rotate(180))
        alpha_over(img, top, (0, 2))
        alpha_over(img, front, (11, 7))

    elif data & 7 == 4: # west
        side = self.transform_image_side(side_r)
        alpha_over(img, side.transpose(Image.FLIP_LEFT_RIGHT), (11, 7))
        front = self.transform_image_side(front)
        alpha_over(img, front, (1, 7))
        top = self.transform_image_top(top.rotate(270))
        alpha_over(img, top, (0, 2))

    elif data & 7 == 5: # east
        back = self.transform_image_side(back)
        side = self.transform_image_side(side_l).transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, side, (11, 7))
        alpha_over(img, back, (1, 7))
        top = self.transform_image_top(top.rotate(270))
        alpha_over(img, top, (0, 2))
        
    else: # just in case
        img = None

    return img

# redstone wire
# uses pseudo-ancildata found in iterate.c
@material(blockid=55, data=list(range(128)), transparent=True)
def wire(self, blockid, data):

    if data & 0b1000000 == 64: # powered redstone wire
        redstone_wire_t = self.assetLoader.load_image_texture("assets/minecraft/textures/block/redstone_dust_line0.png").rotate(90)
        redstone_wire_t = self.tint_texture(redstone_wire_t,(255,0,0))

        redstone_cross_t = self.assetLoader.load_image_texture("assets/minecraft/textures/block/redstone_dust_dot.png")
        redstone_cross_t = self.tint_texture(redstone_cross_t,(255,0,0))

        
    else: # unpowered redstone wire
        redstone_wire_t = self.assetLoader.load_image_texture("assets/minecraft/textures/block/redstone_dust_line0.png").rotate(90)
        redstone_wire_t = self.tint_texture(redstone_wire_t,(48,0,0))
        
        redstone_cross_t = self.assetLoader.load_image_texture("assets/minecraft/textures/block/redstone_dust_dot.png")
        redstone_cross_t = self.tint_texture(redstone_cross_t,(48,0,0))

    # generate an image per redstone direction
    branch_top_left = redstone_cross_t.copy()
    ImageDraw.Draw(branch_top_left).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_top_left).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_top_left).rectangle((0,11,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    
    branch_top_right = redstone_cross_t.copy()
    ImageDraw.Draw(branch_top_right).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_top_right).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_top_right).rectangle((0,11,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    
    branch_bottom_right = redstone_cross_t.copy()
    ImageDraw.Draw(branch_bottom_right).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_bottom_right).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_bottom_right).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    branch_bottom_left = redstone_cross_t.copy()
    ImageDraw.Draw(branch_bottom_left).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_bottom_left).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_bottom_left).rectangle((0,11,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
            
    # generate the bottom texture
    if data & 0b111111 == 0:
        bottom = redstone_cross_t.copy()

    # see iterate.c for where these masks come from
    has_x = (data & 0b1010) > 0
    has_z = (data & 0b0101) > 0
    if has_x and has_z:
        bottom = redstone_cross_t.copy()
        if has_x:
            alpha_over(bottom, redstone_wire_t.copy())
        if has_z:
            alpha_over(bottom, redstone_wire_t.copy().rotate(90))

    else:
        if has_x:
            bottom = redstone_wire_t.copy()
        elif has_z:
            bottom = redstone_wire_t.copy().rotate(90)
        elif data & 0b1111 == 0: 
            bottom = redstone_cross_t.copy()

    # check for going up redstone wire
    if data & 0b100000 == 32:
        side1 = redstone_wire_t.rotate(90)
    else:
        side1 = None
        
    if data & 0b010000 == 16:
        side2 = redstone_wire_t.rotate(90)
    else:
        side2 = None
        
    img = self.build_full_block(None,side1,side2,None,None,bottom)

    return img

# signposts
@material(blockid=[63,11401,11402,11403,11404,11405,11406], data=list(range(16)), transparent=True)
def signpost(self, blockid, data):

    # first rotations
    if self.rotation == 1:
        data = (data + 4) % 16
    elif self.rotation == 2:
        data = (data + 8) % 16
    elif self.rotation == 3:
        data = (data + 12) % 16
    
    sign_texture = {
        # (texture on sign, texture on stick)
        63: ("oak_planks.png", "oak_log.png"),
        11401: ("oak_planks.png", "oak_log.png"),
        11402: ("spruce_planks.png", "spruce_log.png"),
        11403: ("birch_planks.png", "birch_log.png"),
        11404: ("jungle_planks.png", "jungle_log.png"),
        11405: ("acacia_planks.png", "acacia_log.png"),
        11406: ("dark_oak_planks.png", "dark_oak_log.png"),
    }
    texture_path, texture_stick_path = ["assets/minecraft/textures/block/" + x for x in sign_texture[blockid]]
    
    texture = self.assetLoader.load_image_texture(texture_path).copy()
    
    # cut the planks to the size of a signpost
    ImageDraw.Draw(texture).rectangle((0,12,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # If the signpost is looking directly to the image, draw some 
    # random dots, they will look as text.
    if data in (0,1,2,3,4,5,15):
        for i in range(15):
            x = randint(4,11)
            y = randint(3,7)
            texture.putpixel((x,y),(0,0,0,255))

    # Minecraft uses wood texture for the signpost stick
    texture_stick = self.assetLoader.load_image_texture(texture_stick_path)
    texture_stick = texture_stick.resize((12,12), Image.ANTIALIAS)
    ImageDraw.Draw(texture_stick).rectangle((2,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))

    img = Image.new("RGBA", (24,24), self.bgcolor)

    #         W                N      ~90       E                   S        ~270
    angles = (330.,345.,0.,15.,30.,55.,95.,120.,150.,165.,180.,195.,210.,230.,265.,310.)
    angle = math.radians(angles[data])
    post = self.transform_image_angle(texture, angle)

    # choose the position of the "3D effect"
    incrementx = 0
    if data in (1,6,7,8,9,14):
        incrementx = -1
    elif data in (3,4,5,11,12,13):
        incrementx = +1

    alpha_over(img, texture_stick,(11, 8),texture_stick)
    # post2 is a brighter signpost pasted with a small shift,
    # gives to the signpost some 3D effect.
    post2 = ImageEnhance.Brightness(post).enhance(1.2)
    alpha_over(img, post2,(incrementx, -3),post2)
    alpha_over(img, post, (0,-2), post)

    return img

# wall signs
@material(blockid=[68,11407,11408,11409,11410,11411,11412], data=[2, 3, 4, 5], transparent=True)
def wall_sign(self, blockid, data): # wall sign

    # first rotations
    if self.rotation == 1:
        if data == 2: data = 5
        elif data == 3: data = 4
        elif data == 4: data = 2
        elif data == 5: data = 3
    elif self.rotation == 2:
        if data == 2: data = 3
        elif data == 3: data = 2
        elif data == 4: data = 5
        elif data == 5: data = 4
    elif self.rotation == 3:
        if data == 2: data = 4
        elif data == 3: data = 5
        elif data == 4: data = 3
        elif data == 5: data = 2
    
    sign_texture = {
        68: "oak_planks.png",
        11407: "oak_planks.png",
        11408: "spruce_planks.png",
        11409: "birch_planks.png",
        11410: "jungle_planks.png",
        11411: "acacia_planks.png",
        11412: "dark_oak_planks.png",
    }
    texture_path = "assets/minecraft/textures/block/" + sign_texture[blockid]
    texture = self.assetLoader.load_image_texture(texture_path).copy()
    # cut the planks to the size of a signpost
    ImageDraw.Draw(texture).rectangle((0,12,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # draw some random black dots, they will look as text
    """ don't draw text at the moment, they are used in blank for decoration
    
    if data in (3,4):
        for i in range(15):
            x = randint(4,11)
            y = randint(3,7)
            texture.putpixel((x,y),(0,0,0,255))
    """
    
    img = Image.new("RGBA", (24,24), self.bgcolor)

    incrementx = 0
    if data == 2:  # east
        incrementx = +1
        sign = self.build_full_block(None, None, None, None, texture)
    elif data == 3:  # west
        incrementx = -1
        sign = self.build_full_block(None, texture, None, None, None)
    elif data == 4:  # north
        incrementx = +1
        sign = self.build_full_block(None, None, texture, None, None)
    elif data == 5:  # south
        incrementx = -1
        sign = self.build_full_block(None, None, None, texture, None)

    sign2 = ImageEnhance.Brightness(sign).enhance(1.2)
    alpha_over(img, sign2,(incrementx, 2),sign2)
    alpha_over(img, sign, (0,3), sign)

    return img

# nether and normal fences
# uses pseudo-ancildata found in iterate.c
@material(blockid=[85, 188, 189, 190, 191, 192, 113], data=list(range(16)), transparent=True, nospawn=True)
def fence(self, blockid, data):
    # no need for rotations, it uses pseudo data.
    # create needed images for Big stick fence
    if blockid == 85: # normal fence
        fence_top = self.assetLoader.load_image_texture("assets/minecraft/textures/block/oak_planks.png").copy()
        fence_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/oak_planks.png").copy()
        fence_small_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/oak_planks.png").copy()
    elif blockid == 188: # spruce fence
        fence_top = self.assetLoader.load_image_texture("assets/minecraft/textures/block/spruce_planks.png").copy()
        fence_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/spruce_planks.png").copy()
        fence_small_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/spruce_planks.png").copy()
    elif blockid == 189: # birch fence
        fence_top = self.assetLoader.load_image_texture("assets/minecraft/textures/block/birch_planks.png").copy()
        fence_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/birch_planks.png").copy()
        fence_small_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/birch_planks.png").copy()
    elif blockid == 190: # jungle fence
        fence_top = self.assetLoader.load_image_texture("assets/minecraft/textures/block/jungle_planks.png").copy()
        fence_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/jungle_planks.png").copy()
        fence_small_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/jungle_planks.png").copy()
    elif blockid == 191: # big/dark oak fence
        fence_top = self.assetLoader.load_image_texture("assets/minecraft/textures/block/dark_oak_planks.png").copy()
        fence_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/dark_oak_planks.png").copy()
        fence_small_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/dark_oak_planks.png").copy()
    elif blockid == 192: # acacia oak fence
        fence_top = self.assetLoader.load_image_texture("assets/minecraft/textures/block/acacia_planks.png").copy()
        fence_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/acacia_planks.png").copy()
        fence_small_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/acacia_planks.png").copy()
    else: # netherbrick fence
        fence_top = self.assetLoader.load_image_texture("assets/minecraft/textures/block/nether_bricks.png").copy()
        fence_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/nether_bricks.png").copy()
        fence_small_side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/nether_bricks.png").copy()

    # generate the textures of the fence
    ImageDraw.Draw(fence_top).rectangle((0,0,5,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_top).rectangle((10,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_top).rectangle((0,0,15,5),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_top).rectangle((0,10,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    ImageDraw.Draw(fence_side).rectangle((0,0,5,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_side).rectangle((10,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # Create the sides and the top of the big stick
    fence_side = self.transform_image_side(fence_side)
    fence_other_side = fence_side.transpose(Image.FLIP_LEFT_RIGHT)
    fence_top = self.transform_image_top(fence_top)

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = fence_side.split()[3]
    fence_side = ImageEnhance.Brightness(fence_side).enhance(0.9)
    fence_side.putalpha(sidealpha)
    othersidealpha = fence_other_side.split()[3]
    fence_other_side = ImageEnhance.Brightness(fence_other_side).enhance(0.8)
    fence_other_side.putalpha(othersidealpha)

    # Compose the fence big stick
    fence_big = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(fence_big,fence_side, (5,4),fence_side)
    alpha_over(fence_big,fence_other_side, (7,4),fence_other_side)
    alpha_over(fence_big,fence_top, (0,0),fence_top)
    
    # Now render the small sticks.
    # Create needed images
    ImageDraw.Draw(fence_small_side).rectangle((0,0,15,0),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((0,4,15,6),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((0,10,15,16),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # Create the sides and the top of the small sticks
    fence_small_side = self.transform_image_side(fence_small_side)
    fence_small_other_side = fence_small_side.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = fence_small_other_side.split()[3]
    fence_small_other_side = ImageEnhance.Brightness(fence_small_other_side).enhance(0.9)
    fence_small_other_side.putalpha(sidealpha)
    sidealpha = fence_small_side.split()[3]
    fence_small_side = ImageEnhance.Brightness(fence_small_side).enhance(0.9)
    fence_small_side.putalpha(sidealpha)

    # Create img to compose the fence
    img = Image.new("RGBA", (24,24), self.bgcolor)

    # Position of fence small sticks in img.
    # These postitions are strange because the small sticks of the 
    # fence are at the very left and at the very right of the 16x16 images
    pos_top_left = (2,3)
    pos_top_right = (10,3)
    pos_bottom_right = (10,7)
    pos_bottom_left = (2,7)
    
    # +x axis points top right direction
    # +y axis points bottom right direction
    # First compose small sticks in the back of the image, 
    # then big stick and thecn small sticks in the front.

    if (data & 0b0001) == 1:
        alpha_over(img,fence_small_side, pos_top_left,fence_small_side)                # top left
    if (data & 0b1000) == 8:
        alpha_over(img,fence_small_other_side, pos_top_right,fence_small_other_side)    # top right
        
    alpha_over(img,fence_big,(0,0),fence_big)
        
    if (data & 0b0010) == 2:
        alpha_over(img,fence_small_other_side, pos_bottom_left,fence_small_other_side)      # bottom left    
    if (data & 0b0100) == 4:
        alpha_over(img,fence_small_side, pos_bottom_right,fence_small_side)                  # bottom right
    
    return img

# huge brown and red mushroom
@material(blockid=[99,100], data= list(range(11)) + [14,15], solid=True)
def huge_mushroom(self, blockid, data):
    # rotation
    if self.rotation == 1:
        if data == 1: data = 3
        elif data == 2: data = 6
        elif data == 3: data = 9
        elif data == 4: data = 2
        elif data == 6: data = 8
        elif data == 7: data = 1
        elif data == 8: data = 4
        elif data == 9: data = 7
    elif self.rotation == 2:
        if data == 1: data = 9
        elif data == 2: data = 8
        elif data == 3: data = 7
        elif data == 4: data = 6
        elif data == 6: data = 4
        elif data == 7: data = 3
        elif data == 8: data = 2
        elif data == 9: data = 1
    elif self.rotation == 3:
        if data == 1: data = 7
        elif data == 2: data = 4
        elif data == 3: data = 1
        elif data == 4: data = 2
        elif data == 6: data = 8
        elif data == 7: data = 9
        elif data == 8: data = 6
        elif data == 9: data = 3

    # texture generation
    if blockid == 99: # brown
        cap = self.assetLoader.load_image_texture("assets/minecraft/textures/block/brown_mushroom_block.png")
    else: # red
        cap = self.assetLoader.load_image_texture("assets/minecraft/textures/block/red_mushroom_block.png")

    stem = self.assetLoader.load_image_texture("assets/minecraft/textures/block/mushroom_stem.png")
    porous = self.assetLoader.load_image_texture("assets/minecraft/textures/block/mushroom_block_inside.png")
    
    if data == 0: # fleshy piece
        img = self.build_full_block(porous, None, None, porous, porous)

    if data == 1: # north-east corner
        img = self.build_full_block(cap, None, None, cap, porous)

    if data == 2: # east side
        img = self.build_full_block(cap, None, None, porous, cap)

    if data == 3: # south-east corner
        img = self.build_full_block(cap, None, None, porous, cap)

    if data == 4: # north side
        img = self.build_full_block(cap, None, None, cap, porous)

    if data == 5: # top piece
        img = self.build_full_block(cap, None, None, porous, porous)

    if data == 6: # south side
        img = self.build_full_block(cap, None, None, cap, porous)

    if data == 7: # north-west corner
        img = self.build_full_block(cap, None, None, cap, cap)

    if data == 8: # west side
        img = self.build_full_block(cap, None, None, porous, cap)

    if data == 9: # south-west corner
        img = self.build_full_block(cap, None, None, porous, cap)

    if data == 10: # stem
        img = self.build_full_block(porous, None, None, stem, stem)

    if data == 14: # all cap
        img = self.build_block(cap,cap)

    if data == 15: # all stem
        img = self.build_block(stem,stem)

    return img

# iron bars and glass pane
# TODO glass pane is not a sprite, it has a texture for the side,
# at the moment is not used
@material(blockid=[101,102, 160], data=list(range(256)), transparent=True, nospawn=True)
def panes(self, blockid, data):
    # no rotation, uses pseudo data
    if blockid == 101:
        # iron bars
        t = self.assetLoader.load_image_texture("assets/minecraft/textures/block/iron_bars.png")
    elif blockid == 160:
        t = self.assetLoader.load_image_texture("assets/minecraft/textures/block/%s_stained_glass.png" % color_map[data & 0xf])
    else:
        # glass panes
        t = self.assetLoader.load_image_texture("assets/minecraft/textures/block/glass.png")
    left = t.copy()
    right = t.copy()

    # generate the four small pieces of the glass pane
    ImageDraw.Draw(right).rectangle((0,0,7,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(left).rectangle((8,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    
    up_left = self.transform_image_side(left)
    up_right = self.transform_image_side(right).transpose(Image.FLIP_TOP_BOTTOM)
    dw_right = self.transform_image_side(right)
    dw_left = self.transform_image_side(left).transpose(Image.FLIP_TOP_BOTTOM)

    # Create img to compose the texture
    img = Image.new("RGBA", (24,24), self.bgcolor)

    # +x axis points top right direction
    # +y axis points bottom right direction
    # First compose things in the back of the image, 
    # then things in the front.

    # the lower 4 bits encode color, the upper 4 encode adjencies
    data = data >> 4

    if (data & 0b0001) == 1 or data == 0:
        alpha_over(img,up_left, (6,3),up_left)    # top left
    if (data & 0b1000) == 8 or data == 0:
        alpha_over(img,up_right, (6,3),up_right)  # top right
    if (data & 0b0010) == 2 or data == 0:
        alpha_over(img,dw_left, (6,3),dw_left)    # bottom left    
    if (data & 0b0100) == 4 or data == 0:
        alpha_over(img,dw_right, (6,3),dw_right)  # bottom right

    return img

# lilypad
# At the moment of writing this lilypads has no ancil data and their
# orientation depends on their position on the map. So it uses pseudo
# ancildata.
@material(blockid=111, data=list(range(4)), transparent=True)
def lilypad(self, blockid, data):
    t = self.assetLoader.load_image_texture("assets/minecraft/textures/block/lily_pad.png").copy()
    if data == 0:
        t = t.rotate(180)
    elif data == 1:
        t = t.rotate(270)
    elif data == 2:
        t = t
    elif data == 3:
        t = t.rotate(90)

    return self.build_full_block(None, None, None, None, None, t)

# brewing stand
# TODO this is a place holder, is a 2d image pasted
@material(blockid=117, data=list(range(5)), transparent=True)
def brewing_stand(self, blockid, data):
    base = self.assetLoader.load_image_texture("assets/minecraft/textures/block/brewing_stand_base.png")
    img = self.build_full_block(None, None, None, None, None, base)
    t = self.assetLoader.load_image_texture("assets/minecraft/textures/block/brewing_stand.png")
    stand = self.build_billboard(t)
    alpha_over(img,stand,(0,-2))
    return img

# end portal and end_gateway
@material(blockid=[119,209], transparent=True, nodata=True)
def end_portal(self, blockid, data):
    img = Image.new("RGBA", (24,24), self.bgcolor)
    # generate a black texure with white, blue and grey dots resembling stars
    t = Image.new("RGBA", (16,16), (0,0,0,255))
    for color in [(155,155,155,255), (100,255,100,255), (255,255,255,255)]:
        for i in range(6):
            x = randint(0,15)
            y = randint(0,15)
            t.putpixel((x,y),color)
    if blockid == 209: # end_gateway
        return  self.build_block(t, t)
        
    t = self.transform_image_top(t)
    alpha_over(img, t, (0,0), t)

    return img

# end portal frame (data range 8 to get all orientations of filled)
@material(blockid=120, data=list(range(8)), transparent=True)
def end_portal_frame(self, blockid, data):
    # The bottom 2 bits are oritation info but seems there is no
    # graphical difference between orientations
    top = self.assetLoader.load_image_texture("assets/minecraft/textures/block/end_portal_frame_top.png")
    eye_t = self.assetLoader.load_image_texture("assets/minecraft/textures/block/end_portal_frame_eye.png")
    side = self.assetLoader.load_image_texture("assets/minecraft/textures/block/end_portal_frame_side.png")
    img = self.build_full_block((top, 4), None, None, side, side)
    if data & 0x4 == 0x4: # ender eye on it
        # generate the eye
        eye_t = self.assetLoader.load_image_texture("assets/minecraft/textures/block/end_portal_frame_eye.png").copy()
        eye_t_s = self.assetLoader.load_image_texture("assets/minecraft/textures/block/end_portal_frame_eye.png").copy()
        # cut out from the texture the side and the top of the eye
        ImageDraw.Draw(eye_t).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(eye_t_s).rectangle((0,4,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        # trnasform images and paste
        eye = self.transform_image_top(eye_t)
        eye_s = self.transform_image_side(eye_t_s)
        eye_os = eye_s.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, eye_s, (5,5), eye_s)
        alpha_over(img, eye_os, (9,5), eye_os)
        alpha_over(img, eye, (0,0), eye)

    return img

# cobblestone and mossy cobblestone walls, chorus plants, mossy stone brick walls
# one additional bit of data value added for mossy and cobblestone
@material(blockid=[199] + list(range(21000,21013+1)), data=list(range(32)), transparent=True, nospawn=True)
def cobblestone_wall(self, blockid, data):
    walls_id_to_tex = {
          199: "assets/minecraft/textures/block/chorus_plant.png", # chorus plants
        21000: "assets/minecraft/textures/block/andesite.png",
        21001: "assets/minecraft/textures/block/bricks.png",
        21002: "assets/minecraft/textures/block/cobblestone.png",
        21003: "assets/minecraft/textures/block/diorite.png",
        21004: "assets/minecraft/textures/block/end_stone_bricks.png",
        21005: "assets/minecraft/textures/block/granite.png",
        21006: "assets/minecraft/textures/block/mossy_cobblestone.png",
        21007: "assets/minecraft/textures/block/mossy_stone_bricks.png",
        21008: "assets/minecraft/textures/block/nether_bricks.png",
        21009: "assets/minecraft/textures/block/prismarine.png",
        21010: "assets/minecraft/textures/block/red_nether_bricks.png",
        21011: "assets/minecraft/textures/block/red_sandstone.png",
        21012: "assets/minecraft/textures/block/sandstone.png",
        21013: "assets/minecraft/textures/block/stone_bricks.png"
    }
    t = self.assetLoader.load_image_texture(walls_id_to_tex[blockid]).copy()

    wall_pole_top = t.copy()
    wall_pole_side = t.copy()
    wall_side_top = t.copy()
    wall_side = t.copy()
    # _full is used for walls without pole
    wall_side_top_full = t.copy()
    wall_side_full = t.copy()

    # generate the textures of the wall
    ImageDraw.Draw(wall_pole_top).rectangle((0,0,3,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_pole_top).rectangle((12,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_pole_top).rectangle((0,0,15,3),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_pole_top).rectangle((0,12,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    ImageDraw.Draw(wall_pole_side).rectangle((0,0,3,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_pole_side).rectangle((12,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # Create the sides and the top of the pole
    wall_pole_side = self.transform_image_side(wall_pole_side)
    wall_pole_other_side = wall_pole_side.transpose(Image.FLIP_LEFT_RIGHT)
    wall_pole_top = self.transform_image_top(wall_pole_top)

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = wall_pole_side.split()[3]
    wall_pole_side = ImageEnhance.Brightness(wall_pole_side).enhance(0.8)
    wall_pole_side.putalpha(sidealpha)
    othersidealpha = wall_pole_other_side.split()[3]
    wall_pole_other_side = ImageEnhance.Brightness(wall_pole_other_side).enhance(0.7)
    wall_pole_other_side.putalpha(othersidealpha)

    # Compose the wall pole
    wall_pole = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(wall_pole,wall_pole_side, (3,4),wall_pole_side)
    alpha_over(wall_pole,wall_pole_other_side, (9,4),wall_pole_other_side)
    alpha_over(wall_pole,wall_pole_top, (0,0),wall_pole_top)

    # create the sides and the top of a wall attached to a pole
    ImageDraw.Draw(wall_side).rectangle((0,0,15,2),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side).rectangle((0,0,11,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top).rectangle((0,0,11,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top).rectangle((0,11,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    # full version, without pole
    ImageDraw.Draw(wall_side_full).rectangle((0,0,15,2),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top_full).rectangle((0,4,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top_full).rectangle((0,4,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # compose the sides of a wall atached to a pole
    tmp = Image.new("RGBA", (24,24), self.bgcolor)
    wall_side = self.transform_image_side(wall_side)
    wall_side_top = self.transform_image_top(wall_side_top)

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = wall_side.split()[3]
    wall_side = ImageEnhance.Brightness(wall_side).enhance(0.7)
    wall_side.putalpha(sidealpha)

    alpha_over(tmp,wall_side, (0,0),wall_side)
    alpha_over(tmp,wall_side_top, (-5,3),wall_side_top)
    wall_side = tmp
    wall_other_side = wall_side.transpose(Image.FLIP_LEFT_RIGHT)

    # compose the sides of the full wall
    tmp = Image.new("RGBA", (24,24), self.bgcolor)
    wall_side_full = self.transform_image_side(wall_side_full)
    wall_side_top_full = self.transform_image_top(wall_side_top_full.rotate(90))

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = wall_side_full.split()[3]
    wall_side_full = ImageEnhance.Brightness(wall_side_full).enhance(0.7)
    wall_side_full.putalpha(sidealpha)

    alpha_over(tmp,wall_side_full, (4,0),wall_side_full)
    alpha_over(tmp,wall_side_top_full, (3,-4),wall_side_top_full)
    wall_side_full = tmp
    wall_other_side_full = wall_side_full.transpose(Image.FLIP_LEFT_RIGHT)

    # Create img to compose the wall
    img = Image.new("RGBA", (24,24), self.bgcolor)

    # Position wall imgs around the wall bit stick
    pos_top_left = (-5,-2)
    pos_bottom_left = (-8,4)
    pos_top_right = (5,-3)
    pos_bottom_right = (7,4)
    
    # +x axis points top right direction
    # +y axis points bottom right direction
    # There are two special cases for wall without pole.
    # Normal case: 
    # First compose the walls in the back of the image, 
    # then the pole and then the walls in the front.
    if (data == 0b1010) or (data == 0b11010):
        alpha_over(img, wall_other_side_full,(0,2), wall_other_side_full)
    elif (data == 0b0101) or (data == 0b10101):
        alpha_over(img, wall_side_full,(0,2), wall_side_full)
    else:
        if (data & 0b0001) == 1:
            alpha_over(img,wall_side, pos_top_left,wall_side)                # top left
        if (data & 0b1000) == 8:
            alpha_over(img,wall_other_side, pos_top_right,wall_other_side)    # top right

        alpha_over(img,wall_pole,(0,0),wall_pole)
            
        if (data & 0b0010) == 2:
            alpha_over(img,wall_other_side, pos_bottom_left,wall_other_side)      # bottom left    
        if (data & 0b0100) == 4:
            alpha_over(img,wall_side, pos_bottom_right,wall_side)                  # bottom right
    
    return img

@material(blockid=175, data=list(range(16)), transparent=True)
def flower(self, blockid, data):
    double_plant_map = ["sunflower", "lilac", "tall_grass", "large_fern", "rose_bush", "peony", "peony", "peony"]
    plant = double_plant_map[data & 0x7]

    if data & 0x8:
        part = "top"
    else:
        part = "bottom"

    png = "assets/minecraft/textures/block/%s_%s.png" % (plant,part)
    texture = self.assetLoader.load_image_texture(png)
    img = self.build_billboard(texture)

    #sunflower top
    if data == 8:
        bloom_tex = self.assetLoader.load_image_texture("assets/minecraft/textures/block/sunflower_front.png")
        alpha_over(img, bloom_tex.resize((14, 11), Image.ANTIALIAS), (5,5))

    return img

# shulker box
@material(blockid=list(range(219,235)), data=list(range(6)), solid=True, nospawn=True)
def shulker_box(self, blockid, data):
    # first, do the rotation if needed
    data = data & 7
    if self.rotation == 1:
        if data == 2: data = 5
        elif data == 3: data = 4
        elif data == 4: data = 2
        elif data == 5: data = 3
    elif self.rotation == 2:
        if data == 2: data = 3
        elif data == 3: data = 2
        elif data == 4: data = 5
        elif data == 5: data = 4
    elif self.rotation == 3:
        if data == 2: data = 4
        elif data == 3: data = 5
        elif data == 4: data = 3
        elif data == 5: data = 2

    color = color_map[blockid - 219]
    shulker_t = self.assetLoader.load_image_texture("assets/minecraft/textures/entity/shulker/shulker_%s.png" % color).copy()
    w,h = shulker_t.size
    res = w // 4
    # Cut out the parts of the shulker texture we need for the box
    top = shulker_t.crop((res, 0, res * 2, res))
    bottom = shulker_t.crop((res * 2, int(res * 1.75), res * 3, int(res * 2.75)))
    side_top = shulker_t.crop((0, res, res, int(res * 1.75)))
    side_bottom = shulker_t.crop((0, int(res * 2.75), res, int(res * 3.25)))
    side = Image.new('RGBA', (res, res))
    side.paste(side_top, (0, 0), side_top)
    side.paste(side_bottom, (0, res // 2), side_bottom)

    if data == 0: # down
        side = side.rotate(180)
        img = self.build_full_block(bottom, None, None, side, side)
    elif data == 1: # up
        img = self.build_full_block(top, None, None, side, side)
    elif data == 2: # east
        img = self.build_full_block(side, None, None, side.rotate(90), bottom)
    elif data == 3: # west
        img = self.build_full_block(side.rotate(180), None, None, side.rotate(270), top)
    elif data == 4: # north
        img = self.build_full_block(side.rotate(90), None, None, top, side.rotate(270))
    elif data == 5: # south
        img = self.build_full_block(side.rotate(270), None, None, bottom, side.rotate(90))

    return img
