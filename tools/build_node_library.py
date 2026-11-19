"""Rebuild arc_flow/library/tessera_nodes.blend from the code:
   blender --factory-startup -b --python tools/build_node_library.py"""
import os, sys
import bpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
from arc_flow import tessera_nodes as TN
TN.build_library()
print("wrote", TN.LIB_PATH, os.path.getsize(TN.LIB_PATH), "bytes")
