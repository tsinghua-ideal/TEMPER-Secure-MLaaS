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

Install TVM v0.7 from https://github.com/apache/incubator-tvm.git . You can use [TVM Docs][tvm_docs] to install TVM.
[tvm_docs]: https://tvm.apache.org/docs/install/index.html

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

[doc]: https://edp.fortanix.com/docs/installation/guide/
## Evaluation

To run the model partition, you should run `python auto_model_partition.py --model <your_model> --input_size <data_size> --build_dir <path>`. The model will be partitioned into several TVM submodels and the submodels will be compiled into libraries and parameters. The enclave libraries will be stored in the `build_dir` directory.

To run the model inference, you should run the following commands:
```
cd cluster-inference
source environment.sh
./clean.sh
./build.sh
python slave_generator.py <the path of generated models> <the path of target instance dir>

```

## Brute Force 
Brute Force refers to a brute force searching algorithm to find best model partition.
A partition rule is applied, and the best situation of complexity is $O(n)$ while the worst is $O(2^n)$.
if 
$Latency_1..(n+1)≤Latency_(1..n)+ Latency_(n+1)$
, do not partition.
