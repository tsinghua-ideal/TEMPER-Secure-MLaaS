#!/bin/bash
cd /home/lifabing/sgx/best-partion/inference
s=$PATH
export PATH=$1:$PATH
cargo clean
rm -r ~/.cargo/.package-cache
cargo run
path=target/x86_64-fortanix-unknown-sgx/debug/sgx-demo.sgxs
#path=$1
for j in {0..20};
do
	ftxsgx-runner ${path}
done
export PATH=$s
