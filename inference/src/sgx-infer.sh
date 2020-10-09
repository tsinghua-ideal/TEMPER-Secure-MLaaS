#!/bin/bash
dec2hex(){
    printf "%x" $1
}

cd /home/lifabing/sgx/best-partion/inference
cp Cargo.toml.copy Cargo.toml
hp=$(dec2hex $2)
sed -i "s/tobereplaced/$hp/g" Cargo.toml
s=$PATH
export PATH=$1:$PATH
cargo clean
cargo run 2>>/dev/null
path=target/x86_64-fortanix-unknown-sgx/debug/sgx-demo.sgxs
for j in {0..20};
do
	ftxsgx-runner ${path}
done
export PATH=$s
