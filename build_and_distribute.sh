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
CURRENT_PLATFORM=$(conda info --json | grep platform | cut -d'"' -f4)
declare -a PYTHON_VERSIONS=("2.7" "3.5" "3.6")
declare -a PLATFORMS=("linux-64" "osx-64")

# Build for all python version
for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"
do
    echo Building $PYTHON_VERSION
    conda build --output-folder $CONDA_BUILD_DIR --python $PYTHON_VERSION .
done

# Build for all architectures
for FILE in `find $CONDA_BUILD_DIR/$CURRENT_PLATFORM -name $CONDA_PACKAGE_NAME'*' -type f`
do
    for PLATFORM in "${PLATFORMS[@]}"
    do
        echo Converting $FILE for platform $PLATFORM
        conda convert --platform $PLATFORM $FILE -o $CONDA_BUILD_DIR
    done
    echo $FILE
done

# Push to conda
for FILE in `find $CONDA_BUILD_DIR -name $CONDA_PACKAGE_NAME'*' -type f`
do
    echo Uploading $FILE
    anaconda upload $FILE --force
done

