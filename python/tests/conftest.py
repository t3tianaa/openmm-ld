import os

import msldplugin

_library = os.environ.get("MSLD_REFERENCE_PLUGIN")
if _library:
    msldplugin.loadReferencePlugin(_library)
