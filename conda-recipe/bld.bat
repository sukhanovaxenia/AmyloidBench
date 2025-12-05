@echo off

:: Install the package
%PYTHON% -m pip install . -vv --no-deps --no-build-isolation
if errorlevel 1 exit 1

:: Verify installation
%PYTHON% -c "import amyloidbench; print(f'AmyloidBench installed successfully')"
if errorlevel 1 exit 1
