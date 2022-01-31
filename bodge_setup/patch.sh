#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir ./dest
cd $SCRIPT_DIR/dest
git clone https://github.com/mila-iqia/COVI-ML
cd ./COVI-ML
# revert to working version
git checkout 19986f7427a7a643eb05fb41e5ed4dd113362cd6

# insert fixed requirements
cp $SCRIPT_DIR/setup.py ./setup.py
cp $SCRIPT_DIR/requirements.txt ./requirements.txt

# install ctt
pip install -e .