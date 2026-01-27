# Installation and Packaging Guide

## For Users

### Installing from PyPI (once published)
```bash
# Basic installation
pip install schrodinger-bridge

# With optional OTT-JAX support
pip install schrodinger-bridge[ott]

# With all optional dependencies
pip install schrodinger-bridge[all]
```

### Installing from Source
```bash
# Clone the repository
git clone https://github.com/yourusername/schrodinger-bridge.git
cd schrodinger-bridge

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e .[all]
```

## For Developers

### Setting Up Development Environment

1. **Clone and enter the repository**
   ```bash
   git clone https://github.com/yourusername/schrodinger-bridge.git
   cd schrodinger-bridge
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode with dev dependencies**
   ```bash
   pip install -e .[dev]
   ```

4. **Verify installation**
   ```bash
   python -c "import schrodinger_bridge; print(schrodinger_bridge.__version__)"
   ```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=schrodinger_bridge --cov-report=html

# Run specific test file
pytest tests/test_solvers.py
```

### Code Formatting
```bash
# Format code with black
black schrodinger_bridge/

# Sort imports
isort schrodinger_bridge/

# Lint with flake8
flake8 schrodinger_bridge/

# Type checking
mypy schrodinger_bridge/
```

## Building and Distributing

### **IMPORTANT: Fix Import Structure First**

Before building the package, you need to fix the import structure. The current codebase has broken imports:

**Current (broken) structure:**
- Files import from `.core.types`, `.core.problem`, `.solvers`, etc.
- But these subdirectories don't exist - all files are at root level

**Two options to fix:**

#### Option 1: Fix the imports to match current structure
Update all imports throughout the codebase:
- Change `from .core.types import ...` to `from .types import ...`
- Change `from .core.problem import ...` to `from .problem import ...`
- Change `from .solvers import ...` to direct imports from solver modules
- Update `__init__.py` similarly

#### Option 2: Reorganize files to match imports
```bash
mkdir schrodinger_bridge/core
mkdir schrodinger_bridge/solvers

# Move files to appropriate locations
mv types.py core/
mv problem.py core/
mv invariants.py core/
mv doob.py solvers/
mv fbsde.py solvers/
# ... etc
```

### Building Distribution Packages

Once imports are fixed:

```bash
# Install build tools
pip install build twine

# Build source distribution and wheel
python -m build

# This creates:
#   dist/schrodinger-bridge-0.1.0.tar.gz
#   dist/schrodinger_bridge-0.1.0-py3-none-any.whl
```

### Testing the Distribution

```bash
# Test installation from the built wheel
pip install dist/schrodinger_bridge-0.1.0-py3-none-any.whl

# Or test the source distribution
pip install dist/schrodinger-bridge-0.1.0.tar.gz
```

### Publishing to PyPI

```bash
# Check the distribution files
twine check dist/*

# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ schrodinger-bridge

# If everything works, upload to production PyPI
twine upload dist/*
```

### Publishing to GitHub

```bash
# Tag the release
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0

# Create release on GitHub and attach distribution files
# - dist/schrodinger-bridge-0.1.0.tar.gz
# - dist/schrodinger_bridge-0.1.0-py3-none-any.whl
```

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError: No module named 'schrodinger_bridge.core'`:
- The import structure hasn't been fixed yet
- Follow the steps in "Fix Import Structure First" section above

### JAX Installation Issues
```bash
# For CPU only
pip install --upgrade "jax[cpu]"

# For CUDA (check JAX docs for correct CUDA version)
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For TPU
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Missing Dependencies
```bash
# Install all dependencies from requirements
pip install -r requirements.txt

# Or install with all optional dependencies
pip install .[all]
```

## Continuous Integration

Consider setting up CI/CD with GitHub Actions:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e .[dev]
    - name: Run tests
      run: |
        pytest --cov=schrodinger_bridge
```

## Documentation

Build documentation locally:

```bash
# Install documentation dependencies
pip install .[docs]

# Build HTML documentation
cd docs
make html

# View documentation
open _build/html/index.html
```
