#!/usr/bin/env python3

import importlib.util

if importlib.util.find_spec("setuptools_scm") is None:
    raise ImportError("setuptools-scm is not installed. Install it by `pip3 install setuptools-scm`")

import os
import subprocess
import sys
from os import path

from setuptools import find_packages, setup
from setuptools_scm.version import get_local_dirty_tag

THIS_DIR = path.dirname(path.abspath(__file__))

UPDATE_SUBMODULES = os.environ.get("QUANTUM_ATTN_BUILD_UPDATE_SUBMODULES", "1") == "1"


def is_git_directory(path="."):
    return subprocess.call(["git", "-C", path, "status"], stderr=subprocess.STDOUT, stdout=open(os.devnull, "w")) == 0


if UPDATE_SUBMODULES:
    if is_git_directory(THIS_DIR):
        print("Updating submodules")
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    else:
        print("Not a git directory. Skipping submodule update.")


def my_local_scheme(version):
    # The following is used to build release packages.
    # Users should never use it.
    local_version = os.getenv("QUANTUM_ATTN_BUILD_LOCAL_VERSION")
    if local_version is None:
        return get_local_dirty_tag(version)
    return f"+{local_version}"


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


setup(
    name="quantum_attn",
    use_scm_version={"write_to": path.join("src", "quantum_attn", "_version.py"), "local_scheme": my_local_scheme},
    package_dir={
        "": "src",
    },
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=fetch_requirements(),
    extras_require={
        # optional dependencies, required by some features
        "all": [],
        # dev dependencies. Install them by `pip3 install 'quantum-attn[dev]'`
        "dev": [
            "pre-commit",
            "pytest>=7.0.0,<8.0.0",  # https://github.com/pytest-dev/pytest/issues/12273
            "expecttest",
            #
            "pandas",
            "llnl-hatchet",
        ],
    },
)
