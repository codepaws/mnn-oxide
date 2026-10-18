import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from package_mnn_artifact import PackageError, package_artifact


VERSION = b"""#define MNN_VERSION_MAJOR 3
#define MNN_VERSION_MINOR 6
#define MNN_VERSION_PATCH 1
"""
HEADERS = (
    "MNN/Interpreter.hpp", "MNN/MNNDefine.h", "MNN/MNNForwardType.h",
    "MNN/Tensor.hpp", "MNN/ImageProcess.hpp", "MNN/Matrix.h",
)


class PackageArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.artifact = self.root / "artifact"
        self.source.mkdir()
        (self.source / "include/MNN").mkdir(parents=True)
        (self.source / "include/MNN/MNNDefine.h").write_bytes(VERSION)
        (self.source / "LICENSE.txt").write_text("MNN license\n")
        (self.source / "third_party/codec").mkdir(parents=True)
        (self.source / "third_party/codec/NOTICE.txt").write_text("Codec notice\n")
        self.artifact.mkdir()
        (self.artifact / "include/MNN").mkdir(parents=True)
        (self.artifact / "lib").mkdir()
        for header in HEADERS:
            dest = self.artifact / "include" / header
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(VERSION if header == "MNN/MNNDefine.h" else b"header\n")
        (self.artifact / "lib/libMNN.a").write_bytes(b"library\n")
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        subprocess.run(["git", "-C", str(self.source), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.source), "config", "user.name", "Fixture"], check=True)
        subprocess.run(["git", "-C", str(self.source), "remote", "add", "origin", "https://example.invalid/MNN.git"], check=True)
        subprocess.run(["git", "-C", str(self.source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.source), "commit", "-qm", "fixture"], check=True)

    def tearDown(self):
        self.temp.cleanup()

    def package(self, **kwargs):
        options = dict(target="x86_64-unknown-linux-gnu", link_mode="static",
                       backends=["cpu", "cuda"], optional_modules=[],
                       build_options={"MNN_BUILD_LLM": False},
                       toolchain={"cmake": "3.30"})
        options.update(kwargs)
        return package_artifact(self.artifact, self.source, **options)

    def test_writes_schema_notices_hash_inventory_and_sums(self):
        manifest = self.package()
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["mnn_version"], "3.6.1")
        self.assertEqual(manifest["source_url"], "https://example.invalid/MNN.git")
        self.assertRegex(manifest["source_commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(manifest["backends"], ["cpu", "cuda"])
        self.assertEqual(manifest["toolchain"]["cmake"], "3.30")
        copied = self.artifact / "notices/third_party/third_party/codec/NOTICE.txt"
        self.assertEqual(copied.read_text(), "Codec notice\n")
        self.assertEqual((self.artifact / "notices/LICENSE-MNN").read_text(), "MNN license\n")
        for relative, digest in manifest["files"].items():
            self.assertEqual(digest, hashlib.sha256((self.artifact / relative).read_bytes()).hexdigest())
        sums = (self.artifact / "SHA256SUMS").read_text()
        self.assertIn("include/MNN/Interpreter.hpp", sums)
        self.assertIn("notices/LICENSE-MNN", sums)
        self.assertNotIn("mnn-oxide-artifact.json", sums)
        self.assertNotIn("SHA256SUMS", sums)
        saved = json.loads((self.artifact / "mnn-oxide-artifact.json").read_text())
        self.assertEqual(saved, manifest)

    def test_rejects_missing_header_library_license_or_version_macro(self):
        (self.artifact / "include/MNN/Tensor.hpp").unlink()
        with self.assertRaisesRegex(PackageError, "Tensor.hpp"):
            self.package()
        (self.artifact / "include/MNN/Tensor.hpp").write_bytes(b"header")
        (self.artifact / "lib/libMNN.a").unlink()
        with self.assertRaisesRegex(PackageError, "expected static MNN library"):
            self.package()
        (self.artifact / "lib/libMNN.a").write_bytes(b"library")
        (self.source / "LICENSE.txt").unlink()
        with self.assertRaisesRegex(PackageError, "missing LICENSE"):
            self.package()
        (self.source / "LICENSE.txt").write_text("license")
        (self.source / "include/MNN/MNNDefine.h").write_text("#define MNN_VERSION_MAJOR 3\n")
        with self.assertRaisesRegex(PackageError, "MNN_VERSION_MINOR"):
            self.package()

    def test_requires_artifact_and_source_versions_to_match(self):
        (self.artifact / "include/MNN/MNNDefine.h").write_text(
            "#define MNN_VERSION_MAJOR 3\n#define MNN_VERSION_MINOR 5\n#define MNN_VERSION_PATCH 0\n")
        with self.assertRaisesRegex(PackageError, "source checkout declares 3.6.1"):
            self.package()

    def test_checks_dynamic_library_pair_and_optional_headers(self):
        with self.assertRaisesRegex(PackageError, "llm/llm.hpp"):
            self.package(optional_modules=["llm"])
        (self.artifact / "lib/libMNN.so").write_bytes(b"dynamic")
        self.package(link_mode="dynamic")
        (self.artifact / "lib/libMNN.so").unlink()
        with self.assertRaisesRegex(PackageError, "expected dynamic MNN library"):
            self.package(link_mode="dynamic")


if __name__ == "__main__":
    unittest.main()
