"""Count MAP01 linedef specials in the freedoom2.wad bundled with vizdoom (docs/freeplay.md, "GRPO on MAP01").

    python scripts/map01_specials.py
"""
import collections, hashlib, struct
from pathlib import Path

import vizdoom

wad = (Path(vizdoom.__file__).parent / "freedoom2.wad").read_bytes()
n, off = struct.unpack_from("<ii", wad, 4)
lumps = [struct.unpack_from("<ii8s", wad, off + 16 * i) for i in range(n)]
names = [x[2].rstrip(b"\0").decode() for x in lumps]
pos, size, _ = lumps[names.index("MAP01") + 2]          # MAP01, THINGS, LINEDEFS, ...
assert names[names.index("MAP01") + 2] == "LINEDEFS"
lines = [struct.unpack_from("<hhhhhhh", wad, pos + 14 * k) for k in range(size // 14)]
specials = collections.Counter(ld[3] for ld in lines if ld[3])
print("freedoom2.wad sha256", hashlib.sha256(wad).hexdigest(), "vizdoom", vizdoom.__version__)
print("linedefs", len(lines), "specials", dict(sorted(specials.items())))
print("exit linedefs (11 S1, 51 S1 secret, 52 W1, 124 W1 secret):",
      [(i, ld[3]) for i, ld in enumerate(lines) if ld[3] in (11, 51, 52, 124)])
