"""Setup script for Schrödinger Bridge library."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README if it exists
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Version is defined in __init__.py
VERSION = "0.1.0"

setup(
    name="schrodinger-bridge",
    version=VERSION,
    author="Seth Graham",
    description="Production-grade Schrödinger Bridge library for JAX with multiple solver methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/schrodinger-bridge",  # Update with actual URL
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    
    # Core dependencies
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
    ],
    
    # Optional dependencies
    extras_require={
        "ott": [
            "ott-jax>=0.4.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.19.0",
            "nbsphinx>=0.8.0",
            "jupyter>=1.0.0",
        ],
        "all": [
            "ott-jax>=0.4.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.19.0",
            "nbsphinx>=0.8.0",
            "jupyter>=1.0.0",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",  # Update if different license
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for PyPI
    keywords=[
        "schrodinger-bridge",
        "optimal-transport",
        "jax",
        "diffusion-models",
        "stochastic-processes",
        "generative-models",
        "quantitative-finance",
        "martingale",
        "options-pricing",
        "machine-learning",
    ],
    
    # Entry points (for CLT)
    # entry_points={
    #     "console_scripts": [
    #         "sb-train=schrodinger_bridge.cli:main",
    #     ],
    # },
    
    # Include package data (e.g., configuration files, data files)
    include_package_data=True,
    
    # Package data to include
    package_data={
        "schrodinger_bridge": [
            "py.typed",  # For type checking support
        ],
    },
    
    # Additional metadata
    project_urls={
        "Bug Reports": "https://github.com/yourusername/schrodinger-bridge/issues",
        "Source": "https://github.com/yourusername/schrodinger-bridge",
        "Documentation": "https://schrodinger-bridge.readthedocs.io",  # Update with actual URL
    },
    
    # Zip safety
    zip_safe=False,
)
