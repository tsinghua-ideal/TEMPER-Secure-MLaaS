#!/bin/bash

git clone --recursive https://github.com/grief8/tvm.git tvm
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

cd tvm
mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make -j4

cd ..
export MACOSX_DEPLOYMENT_TARGET=10.9  # This is required for mac to avoid symbol conflicts with libstdc++
cd python; python setup.py install --user; cd ..
pip3 install --user numpy decorator attrstornado psutil xgboost cloudpickle