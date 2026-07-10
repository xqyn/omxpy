'''
project: sequencing
XQ - Leiden UMC
dev
april 3
'''
import sys
import importlib
import inspect
import hashlib
import time

_last_reload = {
    "obj_name"    : None,
    "module_name" : None,
    "file"        : None,
    "before_hash" : None,
    "after_hash"  : None,
    "timestamp"   : None,
}

def _get_bytecode_hash(obj):
    """Hash from compiled bytecode — reflects what's actually loaded in memory."""
    try:
        if inspect.isfunction(obj):
            return hashlib.md5(obj.__code__.co_code).hexdigest()[:8]
        elif inspect.isclass(obj):
            codes = b""
            for _, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                codes += method.__code__.co_code
            return hashlib.md5(codes).hexdigest()[:8]
    except Exception:
        return None

def _get_source_hash(obj):
    """Hash from source file on disk."""
    try:
        src = inspect.getsource(obj)
        return hashlib.md5(src.encode()).hexdigest()[:8]
    except Exception:
        return None
    
def reload(obj):
    """
    Reload the source module of a class or function and return
    the updated version of that object.

    Usage:
        from dev import reload
        SeqQuery = reload(SeqQuery)
        reload(SeqQuery)
    """
    # 1. Discover origin module from the object itself
    module_name = obj.__module__
    obj_name = obj.__qualname__.split(".")[0]

    # 2. Purge module (and any submodules) from sys.modules cache
    stale = [k for k in sys.modules if k == module_name or k.startswith(module_name + ".")]
    for k in stale:
        del sys.modules[k]

    fresh_module = importlib.import_module(module_name)
    fresh_obj = getattr(fresh_module, obj_name)
    
    # 3. Fresh import
    # Auto-rebind in caller's local namespace
    frame = inspect.currentframe().f_back
    frame.f_globals[obj_name] = fresh_obj  # rebind globally in caller

    # 4. Return the updated object from the reloaded module
    return fresh_obj  # still return it for explicit reassign style

def check(obj):
    """
    Check if a reload actually picked up changes.
    Pass this to reload() to get a before/after diff report.

    Usage:
        SeqQuery = reload(check(SeqQuery))
        # or just
        reload(check(SeqQuery))
    """
    module_name = obj.__module__
    obj_name    = obj.__qualname__.split(".")[0]

    # Snapshot BEFORE reload
    before_hash   = _get_source_hash(obj)
    before_time   = time.strftime("%H:%M:%S")
    module_file   = inspect.getfile(obj)

    print(f"\n[dev.check] ── Reload check for '{obj_name}' ──────────────────")
    print(f"  module   : {module_name}")
    print(f"  file     : {module_file}")
    print(f"  hash(before): {before_hash}  @ {before_time}")

    # Purge cache + fresh import (same as reload())
    stale = [k for k in sys.modules if k == module_name or k.startswith(module_name + ".")]
    for k in stale:
        del sys.modules[k]

    fresh_module = importlib.import_module(module_name)
    fresh_obj    = getattr(fresh_module, obj_name)

    # Snapshot AFTER reload
    after_hash = _get_source_hash(fresh_obj)
    after_time = time.strftime("%H:%M:%S")

    print(f"  hash(after) : {after_hash}  @ {after_time}")

    if before_hash != after_hash:
        print(f"  status   : ✅ CHANGED — reload picked up new changes")
    else:
        print(f"  status   : ⚠️  NO CHANGE — source is identical (did you save the file?)")

    print(f"────────────────────────────────────────────────────────────\n")

    # Re-bind in caller's global namespace
    frame = inspect.currentframe().f_back
    frame.f_globals[obj_name] = fresh_obj

    return fresh_obj


def dev_check():
    print("[dev] check — import dev works ✅")
    print("1")