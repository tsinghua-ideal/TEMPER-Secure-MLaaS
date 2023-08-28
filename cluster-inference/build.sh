PartID=$1
TARGET_NAME=sgx-task-enclave
TARGET_DIR=`pwd`/sgx-task-enclave/target/x86_64-fortanix-unknown-sgx/debug
TARGET=$TARGET_DIR/$TARGET_NAME
TARGET_SGX=$TARGET_DIR/$TARGET_NAME.sgxs
TARGET_SIG=$TARGET_DIR/$TARGET_NAME.sig
KEY=ra-enclave/examples/data/vendor-keys/private_key.pem

# Build and sign enclave
(cd sgx-task-enclave && cargo build --target x86_64-fortanix-unknown-sgx ) && \
ftxsgx-elf2sgxs $TARGET --heap-size 0x10000000 --stack-size 0x800000 --threads 8 \
    --debug --output $TARGET_SGX && \
#sgxs-sign --key $KEY $TARGET_SGX $TARGET_DIR/$TARGET_NAME.sig -d --xfrm 7/0 --isvprodid 0 --isvsvn 0
sgxs-sign --key $KEY $TARGET_SGX $TARGET_SIG -d --xfrm 7/0 --isvprodid 0 --isvsvn 0

# Build client
# (cd attest-client && cargo build --target x86_64-unknown-linux-gnu  --example attest_client --features verbose )

# Build SP
# (cd ra-sp && cargo build --target x86_64-unknown-linux-gnu )

