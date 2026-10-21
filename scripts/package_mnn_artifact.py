#!/usr/bin/env python3
"""Add notices, provenance metadata, and checksums to a built MNN artifact.

This records source/build claims and hashes files. It does not inspect the
native binaries or attest that they came from the recorded source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys


REQUIRED_HEADERS = (
    "MNN/Interpreter.hpp", "MNN/MNNDefine.h", "MNN/MNNForwardType.h",
    "MNN/Tensor.hpp", "MNN/ImageProcess.hpp", "MNN/Matrix.h",
)
MODULE_HEADERS = {
    "llm": ("llm/llm.hpp",),
    "module": ("MNN/expr/Module.hpp", "MNN/expr/Expr.hpp", "MNN/expr/Executor.hpp"),
    "runtime": ("MNN/expr/Executor.hpp",),
}
NOTICE_NAMES = {"license", "notice", "copying"}
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class PackageError(ValueError):
    pass


def _run_git(source: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(source), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackageError(f"cannot read git metadata from MNN source checkout: {exc}") from exc


def _version(source: Path) -> str:
    path = source / "include/MNN/MNNDefine.h"
    if not path.is_file():
        raise PackageError(f"MNN source is missing version header {path}")
    text = path.read_text(encoding="utf-8")
    values: list[int] = []
    for macro in ("MNN_VERSION_MAJOR", "MNN_VERSION_MINOR", "MNN_VERSION_PATCH"):
        matches = re.findall(r"^\s*#\s*define\s+" + macro + r"\s+(\d+)\s*(?:\r?$)", text, re.M)
        if not matches:
            raise PackageError(f"MNN version header is missing numeric macro {macro}")
        if len(set(matches)) != 1:
            raise PackageError(f"MNN version header has conflicting values for {macro}")
        values.append(int(matches[0]))
    return ".".join(map(str, values))


def _expected_library(lib: Path, target: str, mode: str) -> tuple[str, ...]:
    windows = "windows" in target
    if mode == "static":
        return ("libMNN.a", "libmnn.a") if not windows or "gnu" in target else ("MNN.lib", "mnn.lib")
    if windows:
        imports = ("libMNN.dll.a", "libmnn.dll.a") if "gnu" in target else ("MNN.lib", "mnn.lib")
        dlls = ("MNN.dll", "mnn.dll")
        if not any((lib / n).is_file() for n in imports) or not any((lib / n).is_file() for n in dlls):
            raise PackageError(f"artifact lib/ lacks expected dynamic import library and DLL for {target}")
        return ()
    return ("libMNN.dylib", "libmnn.dylib") if ("apple" in target or "darwin" in target) else ("libMNN.so", "libmnn.so")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_artifact(artifact: Path, source: Path, *, target: str, link_mode: str,
                     backends: list[str], optional_modules: list[str],
                     build_options: dict, toolchain: dict) -> dict:
    artifact = artifact.resolve()
    source = source.resolve()
    if not artifact.is_dir() or not source.is_dir():
        raise PackageError("artifact and source paths must be existing directories")
    if link_mode not in ("static", "dynamic"):
        raise PackageError("link_mode must be static or dynamic")
    unknown = sorted(set(optional_modules) - MODULE_HEADERS.keys())
    if unknown:
        raise PackageError("unknown optional module(s): " + ", ".join(unknown))
    if not isinstance(build_options, dict) or not isinstance(toolchain, dict):
        raise PackageError("build_options and toolchain must be JSON objects")

    include, lib = artifact / "include", artifact / "lib"
    if not include.is_dir() or not lib.is_dir():
        raise PackageError("artifact must contain include/ and lib/ directories")
    headers = list(REQUIRED_HEADERS)
    for module in optional_modules:
        headers.extend(MODULE_HEADERS[module])
    if "llm" in optional_modules and not (include / "llm/llm.hpp").is_file():
        llm_headers = source / "transformers/llm/engine/include/llm"
        if not (llm_headers / "llm.hpp").is_file():
            raise PackageError("MNN source checkout is missing transformers/llm/engine/include/llm/llm.hpp")
        shutil.copytree(llm_headers, include / "llm", dirs_exist_ok=True)
    for name in headers:
        if not (include / name).is_file():
            raise PackageError(f"artifact include/ is missing required header {name}")
    libnames = _expected_library(lib, target, link_mode)
    if libnames and not any((lib / name).is_file() for name in libnames):
        raise PackageError(f"artifact lib/ lacks expected {link_mode} MNN library for {target}")
    # The Rust sys crate links with the lowercase mnn name on Unix. Normalize
    # upstream's common MNN filename while keeping the original as evidence.
    aliases = {
        ("static", "linux"): ("libMNN.a", "libmnn.a"),
        ("dynamic", "linux"): ("libMNN.so", "libmnn.so"),
        ("static", "darwin"): ("libMNN.a", "libmnn.a"),
        ("dynamic", "darwin"): ("libMNN.dylib", "libmnn.dylib"),
    }
    platform = "darwin" if ("apple" in target or "darwin" in target) else "linux"
    alias_source, alias_target = aliases.get((link_mode, platform), (None, None))
    if alias_source and (lib / alias_source).is_file() and not (lib / alias_target).exists():
        shutil.copy2(lib / alias_source, lib / alias_target)

    version = _version(source)
    artifact_version = _version(artifact)
    if artifact_version != version:
        raise PackageError(f"artifact headers declare MNN {artifact_version}, source checkout declares {version}")
    license_path = next(
        (source / name for name in ("LICENSE", "LICENSE.txt", "LICENSE.md") if (source / name).is_file()),
        None,
    )
    if license_path is None:
        raise PackageError("MNN source checkout is missing LICENSE, LICENSE.txt, or LICENSE.md")
    commit = _run_git(source, "rev-parse", "HEAD")
    if not COMMIT_RE.fullmatch(commit):
        raise PackageError("MNN source checkout did not provide a full 40-character commit")
    source_url = _run_git(source, "config", "--get", "remote.origin.url")
    if not source_url:
        raise PackageError("MNN source checkout has no origin URL")

    notices = artifact / "notices"
    notices.mkdir(exist_ok=True)
    shutil.copy2(license_path, notices / "LICENSE-MNN")
    notice_paths = {"notices/LICENSE-MNN"}
    ignored = {".git", "build", "out", "cmake-build-debug", "cmake-build-release"}
    for root, dirs, files in os.walk(source):
        dirs[:] = sorted(d for d in dirs if d not in ignored)
        for filename in sorted(files):
            stem = Path(filename).stem.casefold()
            if not (stem in NOTICE_NAMES or stem.startswith(("license-", "notice-", "copying-"))) or (Path(root) / filename) == license_path:
                continue
            src_file = Path(root) / filename
            relative = src_file.relative_to(source)
            dest_rel = PurePosixPath("notices/third_party") / PurePosixPath(relative.as_posix())
            dest = artifact.joinpath(*dest_rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest)
            notice_paths.add(dest_rel.as_posix())

    files = {}
    for path in sorted(artifact.rglob("*")):
        if path.is_symlink():
            raise PackageError(f"artifact contains unsupported symlink: {path.relative_to(artifact)}")
        if path.is_file() and path.name not in {"mnn-oxide-artifact.json", "SHA256SUMS"}:
            rel = path.relative_to(artifact).as_posix()
            files[rel] = _sha256(path)
    manifest = {
        "schema_version": 2,
        "mnn_version": version,
        "source_url": source_url,
        "source_commit": commit.lower(),
        "target": target,
        "link_mode": link_mode,
        "backends": backends,
        "optional_modules": optional_modules,
        "build_options": build_options,
        "toolchain": toolchain,
        "notice_files": sorted(notice_paths),
        "files": files,
    }
    (artifact / "mnn-oxide-artifact.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (artifact / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as stream:
        for name, digest in files.items():
            stream.write(f"{digest}  {name}\n")
    return manifest


def _json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("source_dir", type=Path, help="exact MNN source checkout")
    parser.add_argument("--target", required=True)
    parser.add_argument("--link-mode", required=True, choices=("static", "dynamic"))
    parser.add_argument("--backend", action="append", default=[], dest="backends")
    parser.add_argument("--optional-module", action="append", default=[], dest="optional_modules")
    parser.add_argument("--build-options", type=_json_object, default={})
    parser.add_argument("--toolchain", type=_json_object, default={})
    args = parser.parse_args(argv)
    try:
        package_artifact(args.artifact_dir, args.source_dir, target=args.target,
                         link_mode=args.link_mode, backends=args.backends,
                         optional_modules=args.optional_modules,
                         build_options=args.build_options, toolchain=args.toolchain)
    except (PackageError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
