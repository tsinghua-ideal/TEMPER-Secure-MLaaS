PartID=$1
TARGET_NAME=sgx-task-enclave
TARGET_DIR=`pwd`/sgx-task-enclave/target/x86_64-fortanix-unknown-sgx/debug
TARGET=$TARGET_DIR/$TARGET_NAME.sgxs

# Run enclave with the default runner
ftxsgx-runner --signature coresident $TARGET &

# Run client
#(cd ra-client && cargo run --target x86_64-unknown-linux-gnu -Zfeatures=itarget --features "verbose" --example tls-client -- -e 127.0.0.1:7710 -s 127.0.0.1:1234 -n 0) &
(cd ra-client && cargo run --target x86_64-unknown-linux-gnu  --features "verbose" --example tls-client -- -e 127.0.0.1:7711 -s 127.0.0.1:1234  -n 1) &

# Run SP
(cd ra-sp && cargo run --target x86_64-unknown-linux-gnu  --example tls-sp   --features "verbose")
