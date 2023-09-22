# Secure-MLaaS
A Secure MLaaS Framework based on Intel SGX.
The framework contains two parts: model partition and model inference.


## Preparation

Make sure you have SGX v1 with limited EPC. See https://github.com/intel/linux-sgx .
## Installation 

1. Install the python packages

Install the python packages according to the `requirements.txt` on Python 3.6.9.

```
pip3 install -r requirements.txt
```

Note that the TVM packages should be installed by compiled packages. 

2. Install TVM

Install TVM v0.7 from https://github.com/grief8/tvm.git or https://github.com/apache/incubator-tvm.git . You can use [TVM Docs][tvm_docs] to install TVM.
[tvm_docs]: https://tvm.apache.org/docs/install/index.html

You can also refer to the following commands:
```shell
git clone --recursive https://github.com/grief8/tvm.git tvm
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make -j4

cd ../python; python setup.py install --user; cd ..
```

After the compilation, install the python packages.

1. Prepare the Rust environment

Open a terminal and enter the following command:
```
sudo apt install -y build-essential
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```
This command will download a script and start installing the rustup tool, which will install the latest stable version of Rust. You may be prompted for an administrator password.
If the installation was successful, the following line will appear:
```
Rust is installed now. Great!
```

Then, you should switch the rustup toolchain to nightly. 
Install the nightly version:
`rustup install nightly`
Switch to the nightly version of cargo (nightly-2021-04-15-x86_64-unknown-linux-gnu is recommended):
`rustup default nightly-2021-04-15-x86_64-unknown-linux-gnu`

4. Install Fortanix

Fortanix is a target for Intel SGX which automatically compiles the code into SGX SDK. Install it by its official [doc][doc]. Note that Intel SGX SDK is necessary here.

Then run `rustup component add llvm-tools-preview` to get llvm-ar and llvm-objcopy

[doc]: https://edp.fortanix.com/docs/installation/guide/
## Evaluation

To run the model partition, you should run `python auto_model_partition.py --model <your_model> --input_size <data_size> --build_dir <path>`. The model will be partitioned into several TVM submodels and the submodels will be compiled into libraries and parameters. The enclave libraries will be stored in the `build_dir` directory.

To run the model inference, you should run the following commands:
```
cd cluster-inference
source environment.sh

./clean.sh
./build.sh


# Generate instances
python worker_generator.py <the path of generated models> <the path of target instance dir>

./run.sh <the path of target instance dir>

```

<!-- # Build client
cd attest-client && cargo run --target x86_64-unknown-linux-gnu --features verbose --example attest_client -- -e 127.0.0.1:7710 -s 127.0.0.1:1234 -n 0

# Build SP
cd ra-sp && cargo run --target x86_64-unknown-linux-gnu --example tvm_user  -- -e 127.0.0.1:22000 -n 2

# Build Scheduler
cd scheduler;cargo build --target x86_64-fortanix-unknown-sgx --example scheduler

# Build and sign enclave

sgx-task-enclave is the template of worker enclaves. To generate the configurations and the codes of worker enclaves on demand, we set up a worker generator. The worker generator will generate the enclave code and the enclave library.
The enclave library will be signed by the following command:
python worker_generator.py <model_path> <target_dir>
<!-- (cd sgx-task-enclave && cargo build --target x86_64-fortanix-unknown-sgx ) && \
ftxsgx-elf2sgxs $TARGET --heap-size 0x10000000 --stack-size 0x800000 --threads 8 \
    --debug --output $TARGET_SGX && \
#sgxs-sign --key $KEY $TARGET_SGX $TARGET_DIR/$TARGET_NAME.sig -d --xfrm 7/0 --isvprodid 0 --isvsvn 0
sgxs-sign --key $KEY $TARGET_SGX $TARGET_SIG -d --xfrm 7/0 --isvprodid 0 --isvsvn 0 -->


## Debugging

1. Encounter the warning `Blocking waiting for file lock on package cache`.
   > Run `rm ~/.cargo/.package-cache` and re-build the project to fix it. We could also disable the rust-analyzer to avoid it.
   >
2. Cannot fetch crates
   > change the crate sources.
```
mkdir ~/.cargo/config
cat << EOF >> ~/.cargo/config
[target.x86_64-fortanix-unknown-sgx]
runner = "ftxsgx-runner-cargo"

[source.crates-io]
registry = "https://github.com/rust-lang/crates.io-index"

replace-with = "tuna"

[source.tuna]
registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"

[source.ustc]
registry = "git://mirrors.ustc.edu.cn/crates.io-index"

[source.sjtu]
registry = "https://mirrors.sjtug.sjtu.edu.cn/git/crates.io-index"

[source.rustcc]
registry = "https://code.aliyun.com/rustcc/crates.io-index.git"
EOF
```

3. Feature `edition2021` is required
> Manually add `edition = "2021"` to the `Cargo.toml` of the error packages.
   
