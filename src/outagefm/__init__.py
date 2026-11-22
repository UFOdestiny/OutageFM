"""OutageFM: county-level power-outage forecasting from operational weather forecasts."""

import importlib
import importlib.abc
import importlib.util
import sys


class _LegacyPackageAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve the alias ``outagefield`` (used in stored evaluation pickles) to ``outagefm``."""

    legacy = "outagefield"

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.legacy or fullname.startswith(self.legacy + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("outagefm" + spec.name[len(self.legacy) :])

    def exec_module(self, module):
        pass


if not any(isinstance(finder, _LegacyPackageAlias) for finder in sys.meta_path):
    sys.meta_path.insert(0, _LegacyPackageAlias())
