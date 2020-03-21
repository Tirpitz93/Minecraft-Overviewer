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


# suffixes = ["inner", "outer", "top", "side", "alt", "bottom", "noside", "locked"]

def parsenamespace(ns_string:str):
    if ns_string.find(":") != -1:
        return ns_string.split(":")
    else:
        return "minecraft", ns_string




def unpackTag(tag:dict, block_tags):
    """Creates an array of blocks associated with a given tag"""
    # print("unpacking tag", tag)
    _arr = []
    for block in tag["values"]:
        if block.startswith("#"):
            # print("found tag", block)
            _arr += unpackTag(block_tags[block[1:]], block_tags)
        else:
            _arr.append(block)
    return _arr




def generate_blocks(jar_file):
    """Unpacks block data from jar file"""
    _blocks = {}
    _block_tags = {}
    _block_models = {}

    zf = zipfile.ZipFile(jar_file, 'r')
    try:
        lst = zf.infolist()
        # print(lst)
        for zi in lst:
            fn = zi.filename

            if fn.startswith(blockstate_path):
                namespace = fn.split("/")[1]
                json_f = json.loads(zf.read(fn))
                json_f["tags"]= []

                # print(namespace)
                # _blocks[namespace] ={}
                _id = "{0}:{1}".format(namespace,os.path.splitext(ntpath.basename(fn))[0])
                _blocks[_id] = json_f

            if fn.startswith(block_tag_path):
                namespace = fn.split("/")[1]
                json_f = json.loads(zf.read(fn))
                _id ="{0}:{1}".format(namespace,os.path.splitext(ntpath.basename(fn))[0])
                _block_tags[_id] = json_f

            if fn.startswith(block_model_path):
                namespace = fn.split("/")[1]
                json_f = json.loads(zf.read(fn))
                name = os.path.splitext(ntpath.basename(fn))[0]
                _id ="{0}:block/{1}".format(namespace,os.path.splitext(ntpath.basename(fn))[0])
                try:
                    _block_models[_id]= json_f

                except:
                    # print("{0}:{1}".format(namespace,os.path.splitext(ntpath.basename(fn))[0]))
                    pass

        _blocks = scan_tags(_blocks,_block_tags)
        # _blocks = scan_models(_blocks, _block_models)

    finally:
        zf.close()
    return _blocks, _block_models, _block_tags

def scan_tags(blocks:dict, block_tags):
    """Adds tag data to blocks"""
    for k,v in block_tags.items():

        tag_blocks = unpackTag(block_tags[k], block_tags)
        if k =="minecraft:buttons":
            pass
        for block in tag_blocks:

            if "tags" in  blocks[block]:
                # print("block {0} already has some tags".format(block))
                pass
            else:
                blocks[block]["tags"]=[]
            blocks[block]["tags"].append(k)
    return blocks


def scan_models(blocks, block_models):
    for blockName, blockData in blocks.items():
        print(blockName)
        pprint(blockData)
        if "variants" in blockData:
            # for variantName, variantData in blockData["variants"].items():
            #     print("variantName" ,variantName)
            #     print("variant data" , variantData)
            #     if type(variantData) == list: #bedrock?????
            #         pass
            #     else:
            #         print(parsenamespace(variantData["model"]))
            #
            #         modelName = "{0}:{1}".format(*parsenamespace(variantData["model"]))
            #         variantData["modelData"] = block_models[modelName]
            pass
        elif "multipart" in blockData:
            for multipartItem in blockData["multipart"]:
                print("multipartItem", multipartItem)
                if type(multipartItem) == list: #bedrock?????
                    pass
                else:
                    print(parsenamespace(variantData["model"]))

                    modelName = "{0}:{1}".format(*parsenamespace(variantData["model"]))
                    variantData["modelData"] = block_models[modelName]
    pass

def cleanModelName(modelName:str, blockID):
    namespace, blockname = parsenamespace(blockID)
    modelName = modelName.split("/")






_blocks, _block_models, _block_tags = generate_blocks(jar_path)


if __name__ == "__main__":
    with open("blocks.json", "w") as outfile:
        json.dump(_blocks, outfile, indent=4, sort_keys=True)
    with open("models.json", "w") as outfile:
        json.dump(_block_models, outfile, indent=4, sort_keys=True)
    with open("tags.json", "w") as outfile:
        json.dump(_block_tags, outfile,indent=4, sort_keys=True)



