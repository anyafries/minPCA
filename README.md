# minPCA

First, create a Python environment:
```bash
python -m venv venv_minpca
source venv_minpca/bin/activate  # On Windows use `venv_minpca\Scripts\activate`
```

The code is organized as a Python package, and can be installed using `pip`:
```bash
git clone https://github.com/anyafries/minPCA.git
cd minPCA
pip install .
cd ..
```
To install it in editable mode (for modifying the code and seeing the changes immediately) and with developer dependencies (for testing and code formatting), use

```bash
pip install -e ".[dev]"
```