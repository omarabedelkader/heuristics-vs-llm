"""Use a pinned Linux VM with the unchanged 64-bit Pharo image on OAR."""
import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

from ranking_corpus import sha256


# Official Pharo 14 VM, v12.0.5-beta+0.7884d28 (same version as the saved Mac VM).
# The stable URL can move; its checksum must match or the run stops.
LINUX_VM_URL = "https://files.pharo.org/get-files/140/pharo-vm-Linux-x86_64-stable.zip"
LINUX_VM_SHA256 = "5f2d6f3b9bc334d5caa50333869ee61108376f4851e2b96564e108cee57db353"


def can_adapt(runtime):
    return (runtime["os"] == "Darwin" and runtime["architecture"] == "arm64"
            and platform.system() == "Linux" and platform.machine() in ("x86_64", "amd64"))


def ensure_runtime(image):
    mac_vm = image / "pharo-vm/Pharo.app"
    if platform.system() != "Linux" or not mac_vm.exists():
        return
    if platform.machine() not in ("x86_64", "amd64"):
        raise ValueError("The pinned Linux VM supports x86_64; this host needs a compatible Pharo VM")
    original_image = sha256(image / "Pharo.image")
    print("Preparing pinned Linux x86_64 Pharo VM; keeping the original image", flush=True)
    with tempfile.TemporaryDirectory(prefix=".native-vm.", dir=image.parent) as temporary:
        stage = Path(temporary)
        archive = stage / "vm.zip"
        subprocess.run(["curl", "-fL", "--retry", "3", "--connect-timeout", "30",
                        "--speed-limit", "1024", "--speed-time", "120",
                        LINUX_VM_URL, "-o", str(archive)], check=True)
        if sha256(archive) != LINUX_VM_SHA256:
            raise ValueError("Linux VM checksum mismatch; the official stable build may have changed. "
                             "Pin and validate the new build before retrying.")
        vm = stage / "pharo-vm"
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                name = PurePosixPath(entry.filename)
                mode = entry.external_attr >> 16
                if name.is_absolute() or ".." in name.parts or "\\" in entry.filename or stat.S_ISLNK(mode):
                    raise ValueError(f"Unsafe VM ZIP member: {entry.filename}")
                target = vm / entry.filename
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                target.chmod(mode & 0o777)
        for name in ("pharo", "lib/pharo", "lib/libPharoVMCore.so"):
            if not (vm / name).is_file():
                raise ValueError(f"Linux VM archive is incomplete: {name}")
        # Check native dependencies before replacing the bundled runtime.
        subprocess.run([str(vm / "pharo"), "--version"], check=True)
        launcher = '#!/usr/bin/env bash\nDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        for name, options in (("pharo", "--headless "), ("pharo-ui", "")):
            path = stage / name
            path.write_text(launcher + f'exec "$DIR/pharo-vm/pharo" {options}"$@"\n')
            path.chmod(0o755)
        # Keep the old runtime available until the replacement has been validated.
        old = image / "pharo-vm"
        old.rename(stage / "previous-vm")
        try:
            vm.rename(old)
        except OSError:
            (stage / "previous-vm").rename(old)
            raise
        for name in ("pharo", "pharo-ui"):
            (stage / name).replace(image / name)
        if sha256(image / "Pharo.image") != original_image:
            raise ValueError("Pharo image changed during VM preparation")
        (image / "runtime-origin.json").write_text(json.dumps({
            "os": "Linux", "architecture": "x86_64", "url": LINUX_VM_URL,
            "archiveSHA256": LINUX_VM_SHA256, "imageSHA256": original_image
        }, indent=2) + "\n")


if __name__ == "__main__":
    try:
        ensure_runtime(Path(sys.argv[1]))
    except (OSError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print(f"Pharo runtime error: {error}", file=sys.stderr)
        sys.exit(1)
