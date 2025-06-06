# setup.py
from pathlib import Path
from setuptools import setup, find_packages

# --------------------------------------------------------------------
# Basic metadata – edit to taste
# --------------------------------------------------------------------
PACKAGE_NAME = "big_brain"            # import big_brain ...
VERSION      = "0.1.0"
DESCRIPTION  = "Big-Brain deep-learning framework (AE / Transformer / AD head)"
AUTHOR       = "Spieterman"
AUTHOR_EMAIL = "you@example.com"      # optional
PY_MIN       = ">=3.8"

# --------------------------------------------------------------------
# Helper: parse README as long_description if it exists
# --------------------------------------------------------------------
readme_path = Path(__file__).with_name("README.md")
long_descr  = readme_path.read_text(encoding="utf-8") if readme_path.exists() else DESCRIPTION

# --------------------------------------------------------------------
# Runtime dependencies
#   • Keep them broad and GPU-agnostic (users install the CUDA-specific
#     Torch build they need separately).
# --------------------------------------------------------------------
install_requires = [
    "torch>=2.0",               # user will pick CPU / cu12 wheel
    "torchvision>=0.15",
    "torchaudio>=2.0",
    "pytorch-lightning>=2.2",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "numpy>=1.23",
    "pandas>=1.5",
    "tqdm>=4.60",
    "nibabel>=5.0",             # for neuroimaging volumes
]

# --------------------------------------------------------------------
# “Editable” src/ layout: all packages live in src/
# --------------------------------------------------------------------
setup(
    name             = PACKAGE_NAME,
    version          = VERSION,
    description      = DESCRIPTION,
    long_description = long_descr,
    long_description_content_type="text/markdown",
    author           = AUTHOR,
    author_email     = AUTHOR_EMAIL,
    python_requires  = PY_MIN,

    # *** THIS IS THE KEY LINE for a src/ layout ***
    package_dir      = {"": "src"},
    packages         = find_packages(where="src"),

    install_requires = install_requires,
    include_package_data = True,          # include package data specified in MANIFEST.in
    license          = "MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
