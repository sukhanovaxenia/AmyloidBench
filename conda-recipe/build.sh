#!/bin/bash
set -ex

# Install the package
${PYTHON} -m pip install . -vv --no-deps --no-build-isolation

# Verify installation
${PYTHON} -c "import amyloidbench; print(f'AmyloidBench {amyloidbench.__version__} installed successfully')"
