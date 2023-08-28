cargo build --target x86_64-fortanix-unknown-sgx --features "verbose"
# cargo build --target x86_64-fortanix-unknown-sgx -Zfeatures=itarget --features "verbose"
ftxsgx-elf2sgxs target/x86_64-fortanix-unknown-sgx/debug/scheduler --heap-size 0x2000000 --stack-size 0x20000 --threads 8 --debug --output target/x86_64-fortanix-unknown-sgx/debug/scheduler.sgxs
sgxs-sign --key ../ra-enclave/examples/data/vendor-keys/private_key.pem target/x86_64-fortanix-unknown-sgx/debug/scheduler.sgxs target/x86_64-fortanix-unknown-sgx/debug/scheduler.sig -d --xfrm 7/0 --isvprodid 0 --isvsvn 0