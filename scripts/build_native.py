#!/usr/bin/env python3
"""Recreate the three original SYCL compile/device-link/host-link steps.
Build in a standalone checkout, with the same Python as ComfyUI. No installation
or dependency changes. The optional PIC device-link flag is experimental.
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sysconfig
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (("base", "qwen_sycl_test"), ("fusion", "qwen_sycl_fused"),
           ("fusion_lut", "qwen_sycl_lut"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sycl-home", default=os.environ.get("SYCL_HOME"),
                        help="Intel oneAPI compiler root containing bin/icpx")
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pic-device-link", action="store_true",
                        help="Experiment: add -fPIC to device-link stage")
    parser.add_argument("--require-no-textrel", action="store_true",
                        help="Fail without publishing built libraries if ELF TEXTREL is present")
    args = parser.parse_args()
    if not args.sycl_home:
        parser.error("Specify --sycl-home or SYCL_HOME")
    import torch
    if not torch.version.xpu:
        parser.error("Use the XPU-enabled Python environment used by ComfyUI")
    torch_root = Path(torch.__file__).resolve().parent
    compiler = Path(args.sycl_home).expanduser().resolve()
    icpx = compiler / "bin" / "icpx"
    cxx = shutil.which(args.cxx)
    readelf = shutil.which("readelf")
    if not icpx.is_file() or not cxx or not readelf:
        parser.error("Missing icpx, host C++ compiler, or readelf")
    includes = [torch_root / "include", torch_root / "include/torch/csrc/api/include",
                compiler / "include", compiler / "include/sycl", Path(sysconfig.get_path("include"))]
    libdirs = [torch_root / "lib", compiler / "lib"]
    for directory in includes + libdirs:
        if not directory.is_dir():
            parser.error(f"Missing include/library directory: {directory}")
    env = dict(os.environ)
    env["SYCL_HOME"] = str(compiler)
    env["PATH"] = str(compiler / "bin") + os.pathsep + env.get("PATH", "")
    # Do not source setvars.sh or replace LD_LIBRARY_PATH: the original environment
    # had a SYCL runtime mismatch when setvars.sh was sourced.
    print(f"PyTorch {torch.__version__}; XPU {torch.version.xpu}; compiler {icpx}", flush=True)
    if str(torch.__version__) != "2.14.1+xpu":
        print("WARNING: this PyTorch version has not been validated by this project", flush=True)
    def run(cmd):
        cmd = [str(v) for v in cmd]
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, env=env, check=True)
    with tempfile.TemporaryDirectory(prefix="qwen-xpu-build-") as tmp:
        tmp = Path(tmp)
        products = []
        textrel = []
        for subdir, name in TARGETS:
            directory = tmp / subdir
            directory.mkdir()
            obj = directory / "gemv.sycl.o"
            dlink = directory / "sycl_dlink.o"
            so = directory / (name + ".so")
            flags = [f"-DTORCH_EXTENSION_NAME={name}", "-DTORCH_API_INCLUDE_EXTENSION_H"]
            for path in includes:
                flags.extend(["-isystem", str(path)])
            flags += ["-fPIC", "-std=c++20"]
            run([icpx, *flags, "-fsycl", "-O3", "-fsycl-targets=spir64", "-sycl-std=2020",
                 f"-fsycl-host-compiler={cxx}",
                 "-fsycl-host-compiler-options=" + shlex.join(flags),
                 "-c", "-x", "c++", ROOT / "native_src" / subdir / "gemv.sycl", "-o", obj])
            run([icpx, obj, "-o", dlink, "-fsycl", "-fsycl-link", "--offload-compress",
                 "-fsycl-targets=spir64", *(["-fPIC"] if args.pic_device_link else [])])
            run([cxx, obj, dlink, "-shared", f"-L{libdirs[0]}", "-lc10", "-lc10_xpu",
                 "-ltorch_cpu", "-ltorch_xpu", "-ltorch", "-ltorch_python",
                 f"-L{libdirs[1]}", "-lsycl", "-o", so])
            if not args.dry_run:
                dynamic = subprocess.check_output([readelf, "-d", str(so)], text=True)
                if "TEXTREL" in dynamic:
                    textrel.append(name)
                products.append(so)
        if args.dry_run:
            return
        if textrel:
            print("WARNING: ELF TEXTREL in " + ", ".join(textrel), flush=True)
            if args.require_no_textrel:
                raise SystemExit("No built libraries installed: TEXTREL check failed")
        # Publish to this checkout only after all three builds succeed.
        dest = ROOT / "lib"
        dest.mkdir(exist_ok=True)
        for so in products:
            shutil.copy2(so, dest / so.name)
            print(f"Built {dest / so.name}", flush=True)
        print("Next: run tests/smoke_native.py in a separate Python process", flush=True)


if __name__ == "__main__":
    main()
