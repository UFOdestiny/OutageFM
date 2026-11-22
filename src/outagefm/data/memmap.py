"""Reopen read-only array mappings when dataset objects enter spawned workers."""

from pathlib import Path

import numpy as np


class ReadOnlyMemmapState:
    """Serialize mapping locations rather than copying the mapped corpus."""

    def __getstate__(self):
        state = self.__dict__.copy()
        mappings = {}
        for name, value in list(state.items()):
            if not isinstance(value, np.memmap) or value.mode != "r" or not value.size:
                continue
            if not (value.flags.c_contiguous or value.flags.f_contiguous):
                continue
            base = value
            while isinstance(base.base, np.ndarray):
                base = base.base
            offset = int(getattr(base, "offset", value.offset)) + value.ctypes.data - base.ctypes.data
            mappings[name] = dict(
                filename=str(Path(value.filename).resolve()),
                dtype=value.dtype,
                mode="r",
                offset=offset,
                shape=value.shape,
                order="F" if value.flags.f_contiguous and not value.flags.c_contiguous else "C",
            )
            del state[name]
        state["_outagefm_readonly_mappings"] = mappings
        return state

    def __setstate__(self, state):
        mappings = state.pop("_outagefm_readonly_mappings", {})
        self.__dict__.update(state)
        for name, specification in mappings.items():
            self.__dict__[name] = np.memmap(**specification)
