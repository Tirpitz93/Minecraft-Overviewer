import zipfile
import json
import ntpath
import os
from pprint import pprint

jar_path = "C:/Users/Lonja/AppData/Roaming/.minecraft/versions/20w11a/20w11a.jar"
block_model_path = "assets/minecraft/models/block"
item_model_path = "assets/minecraft/models/item"

item_tag_path = "data/minecraft/tags/items"
block_tag_path = "data/minecraft/tags/blocks"

blockstate_path = "assets/minecraft/blockstates"
blocks = {}
items = {}
block_tags = {}
item_tags = {}

suffixes = ["inner", "outer", "top", "side", "alt", "bottom", "noside", "locked"]

def parsenamespace(ns_string:str):
    return ns_string.split(":")


def unpackTag(tag:dict):
    print("unpacking tag", tag)
    _arr = []
    for block in tag["values"]:
        if block.startswith("#"):
            print("found tag", block)
            _arr += unpackTag(block_tags[block.split(":")[1]])
        else:
            _arr.append(block)
    return _arr

def generateTextures(jar_file):
    """Unpacks block data from jar file"""
    zf = zipfile.ZipFile(jar_file, 'r')
    try:
        lst = zf.infolist()
        # print(lst)
        for zi in lst:
            fn = zi.filename
            if fn.startswith(block_model_path):

                json_f = json.loads(zf.read(fn))
                json_f["tags"]= []

                blocks[os.path.splitext(ntpath.basename(fn))[0]] = json_f




            if fn.startswith(block_tag_path):
                json_f = json.loads(zf.read(fn))
                block_tags[os.path.splitext(ntpath.basename(fn))[0]] = json_f

            if fn.startswith(blockstate_path):
                json_f = json.loads(zf.read(fn))
                name = os.path.splitext(ntpath.basename(fn))[0]
                try:
                    blocks[name]["states"]= json_f
                except:
                    pass

            # if fn.startswith(item_model_path):
            #
            #     json_f = json.loads(zf.read(fn))
            #
            #     items[os.path.splitext(ntpath.basename(fn))[0]] = json_f
            #
            # if fn.startswith(item_tag_path):
            #
            #     json_f = json.loads(zf.read(fn))
            #
            #     item_tags[os.path.splitext(ntpath.basename(fn))[0]] = json_f


    finally:
            zf.close()
    # print(blocks.keys())
    for k,v in block_tags.items():

        # print("key", k)
        # print("value", v)
        _blocks = unpackTag(block_tags[k])
        if k =="buttons":
            print(_blocks)


        for block in _blocks:
            blockn = block.split(":")[1]
            # print("block", block)
            # print("block", blockn)
            try:
                blocks[blockn]["tags"].append(k)
            except:
                pass

generateTextures(jar_path)

pprint(blocks["spruce_trapdoor_top"]["tags"])

# pprint(blocks["spruce_stairs"])

print(len(blocks))


