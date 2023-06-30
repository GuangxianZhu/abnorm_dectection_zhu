from ctypes import RTLD_LOCAL  # FIXME: Consider availability of RTLD_LOCAL.
import os
import sys
from importlib import import_module


# NOTE: See also cppimport.importer.load_module.
def import_extension(name: str, package: str = None):
    # FIXME: Should os.name is used here?
    if os.name == "nt": # system is Windows
        m = import_module(name, package)  # FIXME: Consider in case of name not found.
        return m
    else:
        # FIXME: Consider availability of sys.{get,set}dlopenflags.
        old_flags = sys.getdlopenflags()
        new_flags = old_flags | RTLD_LOCAL
        sys.setdlopenflags(new_flags)
        m = import_module(name)  # FIXME: Consider in case of name not found.
        sys.setdlopenflags(old_flags)
        return m
