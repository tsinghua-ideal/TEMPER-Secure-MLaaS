#!/bin/bash

rustup target add x86_64-fortanix-unknown-sgx --toolchain nightly
echo "deb https://download.fortanix.com/linux/apt xenial main" | sudo tee -a /etc/apt/sources.list.d/fortanix.list >/dev/null
curl -sSL "https://download.fortanix.com/linux/apt/fortanix.gpg" | sudo -E apt-key add -
sudo apt-get update
sudo apt-get install intel-sgx-dkms

echo "deb https://download.01.org/intel-sgx/sgx_repo/ubuntu $(lsb_release -cs) main" | sudo tee -a /etc/apt/sources.list.d/intel-sgx.list >/dev/null
curl -sSL "https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key" | sudo -E apt-key add -
sudo apt-get update
sudo apt-get install sgx-aesm-service libsgx-aesm-launch-plugin

sudo apt-get install pkg-config libssl-dev protobuf-compiler
cargo install fortanix-sgx-tools sgxs-tools
