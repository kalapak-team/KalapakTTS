import subprocess
import sys
import unittest
from pathlib import Path


DESKTOP_ROOT = Path(__file__).resolve().parents[1]


class ServerIsolationTests(unittest.TestCase):
    def test_desktop_server_starts_without_importing_repository_backend(self) -> None:
        verification = """
import ast
import importlib
import importlib.abc
from pathlib import Path
import sys

desktop_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(desktop_root))

class BackendImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "backend" or fullname.startswith("backend."):
            raise AssertionError("Desktop server attempted to import repository backend: " + fullname)
        return None

sys.meta_path.insert(0, BackendImportBlocker())

for source_path in (desktop_root / "server").glob("*.py"):
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(name == "backend" or name.startswith("backend.") for name in names), source_path

server_module = importlib.import_module("server.main")
assert Path(server_module.__file__).resolve().is_relative_to(desktop_root / "server")
voxcpm_module = importlib.import_module("voxcpm")
assert Path(voxcpm_module.__file__).resolve().is_relative_to(desktop_root / "server" / "vendor")
assert not any(name == "backend" or name.startswith("backend.") for name in sys.modules)
assert not any("VoxCPM" in entry and not str(entry).startswith(str(desktop_root)) for entry in sys.path)
print("desktop server imported without repository backend")
"""
        result = subprocess.run(
            [sys.executable, "-c", verification, str(DESKTOP_ROOT)],
            cwd=DESKTOP_ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("desktop server imported without repository backend", result.stdout)


if __name__ == "__main__":
    unittest.main()
