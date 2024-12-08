#!/bin/bash
# Package specific settings
CONDA_PACKAGE_NAME=gpr1d

# Conda
# Check environment
conda info > /dev/null
if [ $? -ne 0 ]
then
    echo "Unable to run 'conda'. Please install it with your system package manager"; exit 1
fi
conda build --help > /dev/null
anaconda --help > /dev/null 2&>1
if [ $? -ne 0 ]
then
    echo "Unable to run 'anaconda'. Please install it using 'conda install anaconda-client'"; exit 1
fi

CONDA_BUILD_DIR=conda-build
ANACONDA_USER=aaronkho
CURRENT_PLATFORM=$(conda info --json | grep platform | cut -d'"' -f4)
declare -a PYTHON_VERSIONS=("3.8" "3.9" "3.10" "3.11" "3.12")
declare -a PLATFORMS=("linux-64" "osx-64" "osx-arm64")

# Build. This package is pure-python, so only one build necessary
rm -rf $CONDA_BUILD_DIR
conda build --output-folder $CONDA_BUILD_DIR .

# Push to conda
for FILE in `find $CONDA_BUILD_DIR -name $CONDA_PACKAGE_NAME'*' -type f`
do
    echo Uploading $FILE
    anaconda upload $FILE --force --user $ANACONDA_USER
done
