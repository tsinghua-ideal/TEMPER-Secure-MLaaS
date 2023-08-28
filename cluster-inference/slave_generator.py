import sys
import os


def get_config():
    lines = r'''[build]
target = "x86_64-fortanix-unknown-sgx"

[target.x86_64-fortanix-unknown-sgx]
runner = "ftxsgx-runner-cargo"
'''
    return lines


def get_main():
    return r'''extern crate tvm_graph_rt;
extern crate byteorder;
extern crate mbedtls;

use std::net::{TcpListener, TcpStream};
use byteorder::{NetworkEndian, WriteBytesExt};                                                                                              
use ra_enclave::tls_enclave::attestation;
use mbedtls::pk::Pk;
use mbedtls::rng::CtrDrbg;
use mbedtls::ssl::config::{Endpoint, Preset, Transport};
use mbedtls::ssl::{Config, Context, Session};
use mbedtls::x509::Certificate;

#[path = "/home/lww/project/02_Project/01-rust/cluster-inference/support/mod.rs"]
mod support;
use support::entropy::entropy_new;
use support::keys;

use std::{
    convert::TryFrom as _,
    io::{Read as _, Write as _},
    time::{SystemTime, UNIX_EPOCH},
    thread,
    env,
};
use serde_json::{Result, Value};

fn main() {
    let mut thread_vec = vec![];
    let handle = thread::spawn(move ||{
        println!("attestation start");
        let config = include_str!(concat!(env!("PWD"), "/config"));
        let config: Value = serde_json::from_str(config).unwrap();
        let client_address = config["client_address"].as_str().unwrap();
        let sp_address = config["sp_address"].as_str().unwrap();
        // let mut sign_key = attestation(client_address, sp_address, keep_message).unwrap();
        println!("attestation end");
    });
    thread_vec.push(handle);
    let handle = thread::spawn(move ||{
        do_tvm();
    });
    thread_vec.push(handle);
    for handle in thread_vec {
        // Wait for the thread to finish. Returns a result.
        let _ = handle.join().unwrap();
    }
    
 }
 
pub fn keep_message(session:Session){
    let mut sess = session;
    let msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque non placerat risus, et lobortis quam. Mauris velit lorem, elementum id neque a, aliquet tempus turpis. Nam eu congue urna, in semper quam. Ut tristique gravida nunc nec feugiat. Proin tincidunt massa a arcu volutpat, sagittis dignissim velit convallis. Cras ac finibus lorem, nec congue felis. Pellentesque fermentum vitae ipsum sed gravida. Nulla consectetur sit amet erat a pellentesque. Donec non velit sem. Sed eu metus felis. Nullam efficitur consequat ante, ut commodo nisi pharetra consequat. Ut accumsan eget ligula laoreet dictum. Maecenas tristique porta convallis. Suspendisse tempor sodales velit, ac luctus urna varius eu. Ut ultrices urna vestibulum vestibulum euismod. Vivamus eu sapien urna.";
    sess
        .write_u32::<NetworkEndian>(msg.len() as u32)
        .unwrap();
    write!(&mut sess, "{}", msg).unwrap();
}

pub fn do_tvm(){
    env::set_var("TVM_NUM_THREADS", "6");
    let config = include_str!(concat!(env!("PWD"), "/config"));
    let config: Value = serde_json::from_str(config).unwrap();
    let server_address = config["server_address"].as_str().unwrap();
    let client_address = config["client_address"].as_str().unwrap();
    
    let syslib = tvm_graph_rt::SystemLibModule::default();
    let graph_json = include_str!(concat!(env!("OUT_DIR"), "/graph.json"));
    let params_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/params.bin"));
    let params = tvm_graph_rt::load_param_dict(params_bytes).unwrap();
    let graph = tvm_graph_rt::Graph::try_from(graph_json).unwrap();
    let mut exec = tvm_graph_rt::GraphExecutor::new(graph, &syslib).unwrap();
    exec.load_params(params);

    let listener = TcpListener::bind(server_address).unwrap();
    println!("addr: {}", server_address);
    for stream in listener.incoming() {
        let mut socket = TcpStream::connect(client_address).unwrap();
        let mut entropyc = entropy_new();
        let mut rngc = CtrDrbg::new(&mut entropyc, None).unwrap();
        let mut certc = Certificate::from_pem(keys::PEM_CERT).unwrap();
        let mut configc = Config::new(Endpoint::Client, Transport::Stream, Preset::Default);
        configc.set_rng(Some(&mut rngc));
        configc.set_ca_list(Some(&mut *certc), None);
        let mut ctxc = Context::new(&configc).unwrap();
        let mut client_session = ctxc.establish(&mut socket, None).unwrap();

        let mut stream = stream.unwrap();
        let mut entropy = entropy_new();
        let mut rng = CtrDrbg::new(&mut entropy, None).unwrap();
        let mut cert = Certificate::from_pem(keys::PEM_CERT).unwrap();
        let mut key = Pk::from_private_key(keys::PEM_KEY, None).unwrap();
        let mut config = Config::new(Endpoint::Server, Transport::Stream, Preset::Default);
        config.set_rng(Some(&mut rng));
        config.push_cert(&mut *cert, &mut key).unwrap();
        let mut ctx = Context::new(&config).unwrap();
        let mut server_session = ctx.establish(&mut stream, None).unwrap();
        println!("server_session connect!");
        loop {
            if let Err(_) =
                server_session.read(exec.get_input("input").unwrap().data().view().as_mut_slice())
            {
                continue;
            }
            let sy_time = SystemTime::now();
            exec.run();
            client_session.write(exec.get_output(0).unwrap().data().as_slice()).unwrap();
            println!("computing time: {:?}", SystemTime::now().duration_since(sy_time).unwrap().as_micros());
        }
    }
 }

'''


def get_build(model_path):
    return r'''use std::process::Command;
use std::fs;
use std::path::Path;


macro_rules! mf_dir {
    ($p:literal) => {
        concat!(env!("CARGO_MANIFEST_DIR"), $p)
    };
}

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap().to_string();
    fs::create_dir_all(out_dir.clone()).unwrap_or_else(|e| println!("make dir failed：{}", e));
    let origin_path = "''' + model_path + r'''".to_string();
    let p = origin_path.clone();
    for part in fs::read_dir(origin_path).unwrap() {
        let part = part.unwrap().path();
        let md = part.clone();
        let md = md.strip_prefix(p.clone()).unwrap().to_str().unwrap();
        let dest = Path::new(&out_dir.clone()).join(md);
        fs::copy(part, dest).unwrap_or_else(|e| { println!("make dir failed：{}", e); 0}); 
    }

    let sysroot_output = Command::new("rustc")
        .args(&["--print", "sysroot"])
        .output()
        .expect("Failed to get sysroot");
    let sysroot = String::from_utf8(sysroot_output.stdout).unwrap();
    let sysroot = sysroot.trim();
    let mut llvm_tools_path = std::path::PathBuf::from(&sysroot);
    llvm_tools_path.push("lib/rustlib/x86_64-unknown-linux-gnu/bin");

    Command::new("rustup")
        .args(&["component", "add", "llvm-tools-preview"])
        .output()
        .expect("failed to install llvm tools");

    
    std::process::Command::new(llvm_tools_path.join("llvm-objcopy"))
    .arg("--globalize-symbol=__tvm_module_startup")
    .arg("--remove-section=.ctors")
    .arg(&format!("{}/model.o", out_dir))
    .output()
    .expect("could not gloablize startup function");

    std::process::Command::new(llvm_tools_path.join("llvm-ar"))
        .arg("rcs")
        .arg(&format!("{}/libmodel.a", out_dir))
        .arg(&format!("{}/model.o", out_dir))
        .output()
        .expect("failed to package model archive");
    println!("cargo:rustc-link-lib=static=model");
    println!("cargo:rustc-link-search=native={}", out_dir);
    
}
'''


def get_toml():
    return r'''[package]
name = "sample"
version = "0.1.0"
authors = ["fabing <1349212501@qq.com>"]
edition = "2018"

[dependencies]
byteorder = { version = "1.3.2" }
tvm-graph-rt = { path = "/home/lww/project/03_TVM/tvm/rust/tvm-graph-rt" }
mbedtls = {path = "/home/lww/project/03_TVM/rust-mbedtls/mbedtls", default-features = false, features = ["no_std_deps"]}
serde_json = "1.0"
serde = { version = "1.0", features = ["derive"] }
bincode = "1.2.1"
sgx-isa = { version = "0.3.1", features = ["sgxstd"] }
sgx-crypto = { path = "/home/lww/project/02_Project/01-rust/cluster-inference/sgx-crypto" }
ra-common = { path = "/home/lww/project/02_Project/01-rust/cluster-inference/ra-common" }
ra-enclave = { path = "/home/lww/project/02_Project/01-rust/cluster-inference/ra-enclave" }

[package.metadata.fortanix-sgx]
stack-size=0x20000
heap-size=0x3000000
debug=true'''


def get_address(index, model_path):
    return r'''{
    "server_address": "172.16.111.170:'''+ str(32100+index) + r'''",
    "client_address": "172.16.111.236:'''+ str(32100+index+1) + r'''",
    "attestation_address": "127.0.0.1:'''+ str(4240+index) + r'''",
    "sp_address": "127.0.0.1:1310",
    "model": ["resnet18", "mobilenetv1"],
    "model_path": "'''+ str(model_path) + r'''",
    "input_size": [1,3,224,224]
}'''


def my_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    model_path = sys.argv[1] if sys.argv[1] else "/home/lww/project/02_Project/01-rust/cluster-inference/model/resnet50"
    num_models = len(os.listdir(model_path))
    target_dir = sys.argv[2] if sys.argv[2] else "/home/lww/project/02_Project/01-rust//"
    for i in range(num_models):
        root = os.path.join(target_dir, str(i))
        cargo = os.path.join(root, '.cargo')
        src = os.path.join(root, 'src')
        m_path = os.path.join(model_path, str(i))
        print(m_path)
        my_mkdirs(root)
        my_mkdirs(cargo)
        my_mkdirs(src)
        with open(os.path.join(cargo, 'config'), 'w') as config, \
            open(os.path.join(src, 'main.rs'), 'w') as main, \
            open(os.path.join(root, 'build.rs'), 'w') as build, \
            open(os.path.join(root, 'Cargo.toml'), 'w') as toml, \
            open(os.path.join(root, 'config'), 'w') as address:
            config.write(get_config())
            main.write(get_main())
            build.write(get_build(m_path))
            toml.write(get_toml())
            address.write(get_address(i, m_path))
