import importlib

import pytest

MODULES = [
    "jason.__main__",
    "jason.cli",
    "jason.core.events",
    "jason.core.bus",
    "jason.core.clock",
    "jason.core.config",
    "jason.core.state",
    "jason.net.link",
    "jason.net.clocksync",
    "jason.net.protocol",
    "jason.net.admin",
    "jason.log.rawlog",
    "jason.mac.main",
    "jason.brain.main",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
