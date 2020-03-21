import unittest
import os, sys
from generate_textures import *
class testParseNamesapce(unittest.TestCase):
    # @unittest.skip("Broken old garbage, find a newer world")

    def setUp(self):
        self.versiondir = ""
        if "APPDATA" in os.environ and sys.platform.startswith("win"):
            self.versiondir = os.path.join(os.environ['APPDATA'], ".minecraft", "versions")
        elif "HOME" in os.environ:
            # For linux:
            self.versiondir = os.path.join(os.environ['HOME'], ".minecraft", "versions")
            if not os.path.exists( self.versiondir) and sys.platform.startswith("darwin"):
                # For Mac:
                self.versiondir = os.path.join(os.environ['HOME'], "Library",
                                          "Application Support", "minecraft", "versions")
        self.versions= self.scan_versions()


    def scan_versions(self):
        _versions =[]
        "Basic test of the world constructor and regionset constructor"
        if not os.path.exists( self.versiondir):
            raise unittest.SkipTest("test data doesn't exist.  Maybe you need to init/update your submodule?")
        for root, dirs, files in os.walk(self.versiondir):
            for name in files:
                if os.path.splitext(name)[1] ==".jar":
                    print("found jar")
                    print(name)
                    print(root)
                    _versions.append(os.path.join(root, name))

        print(_versions)
        return _versions


    def test_block_generator(self):
        for version in self.versions:
            blocks = generate_blocks(version)


    def test_unpackTag(self):
        pass

    def test_parsenamespace(self):
        self.assertEquals(parsenamespace("minecraft:cobblestone_slab"), ["minecraft","cobblestone_slab"])


        pass






        # regionsets = w.get_regionsets()
        # self.assertEqual(len(regionsets), 3)
        #
        # regionset = regionsets[0]
        # self.assertEqual(regionset.get_region_path(0,0), 'test/data/worlds/exmaple/DIM-1/region/r.0.0.mcr')
        # self.assertEqual(regionset.get_region_path(-1,0), 'test/data/worlds/exmaple/DIM-1/region/r.-1.0.mcr')
        # self.assertEqual(regionset.get_region_path(1,1), 'test/data/worlds/exmaple/DIM-1/region/r.0.0.mcr')
        # self.assertEqual(regionset.get_region_path(35,35), None)
        #
        # # a few random chunks.  reference timestamps fetched with libredstone
        # self.assertEqual(regionset.get_chunk_mtime(0,0), 1316728885)
        # self.assertEqual(regionset.get_chunk_mtime(-1,-1), 1316728886)
        # self.assertEqual(regionset.get_chunk_mtime(5,0), 1316728905)
        # self.assertEqual(regionset.get_chunk_mtime(-22,16), 1316786786)





if __name__ == "__main__":
    unittest.main()