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

use std::net::{TcpListener, TcpStream};
use byteorder::{NetworkEndian, WriteBytesExt};                                                                                              

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

        let mut stream = stream.unwrap();
        println!("server_session connect!");
        loop {
            if let Err(_) =
            stream.read(exec.get_input("input").unwrap().data().view().as_mut_slice())
            {
                continue;
            }
            let sy_time = SystemTime::now();
            exec.run();
            socket.write(exec.get_output(0).unwrap().data().as_slice()).unwrap();
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
tvm-graph-rt = { path = "/home/lifabing/sgx/tvm/rust/tvm-graph-rt" }
serde_json = "1.0"
serde = { version = "1.0", features = ["derive"] }
bincode = "1.2.1"

[package.metadata.fortanix-sgx]
stack-size=0x20000
heap-size=0x3000000
debug=false'''


def get_address(index, model_path):
    return r'''{
    "server_address": "127.0.0.1:'''+ str(32100+index) + r'''",
    "client_address": "127.0.0.1:'''+ str(32100+index+1) + r'''",
    "attestation_address": "127.0.0.1:'''+ str(4240+index) + r'''",
    "sp_address": "127.0.0.1:1310",
    "model": ["resnet18", "mobilenetv1"],
    "model_path": "'''+ str(model_path) + r'''",
    "input_size": [1,3,224,224]
}'''


def get_lock():
    return r'''
[[package]]
name = "platforms"
version = "0.2.1"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "feb3b2b1033b8a60b4da6ee470325f887758c95d5320f52f9ce0df055a55940e"'''

def my_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_client_toml():
    return r'''[package]
name = "sample"
version = "0.1.0"
authors = ["fabing <1349212501@qq.com>"]
edition = "2018"

[dependencies]
byteorder = { version = "1.3.2" }
tvm-graph-rt = { path = "/home/lifabing/sgx/tvm/rust/tvm-graph-rt" }
serde_json = "1.0"
serde = { version = "1.0", features = ["derive"] }
bincode = "1.2.1"
rand = "0.7"

[package.metadata.fortanix-sgx]
stack-size=0x20000
heap-size=0x3000000
debug=false'''


def get_client_main():
    return r'''extern crate tvm_graph_rt;
extern crate byteorder;
extern crate rand;

use std::net::{TcpListener, TcpStream};
use byteorder::{NetworkEndian, WriteBytesExt};                                                                                              
use rand::Rng;
use std::{
    convert::TryFrom as _,
    io::{Read as _, Write as _},
    time::{SystemTime, UNIX_EPOCH},
    thread, slice,
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

pub fn do_tvm(){
    env::set_var("TVM_NUM_THREADS", "6");
    let config = include_str!(concat!(env!("PWD"), "/config"));
    let config: Value = serde_json::from_str(config).unwrap();
    let server_address = config["server_address"].as_str().unwrap();
    let client_address = config["client_address"].as_str().unwrap();
    
    let shape = (1, 3, 224, 224);
    let mut rng =rand::thread_rng();
    let mut ran = vec![];
    for _i in 0..shape.0*shape.1*shape.2*shape.3{
        ran.push(rng.gen::<f32>()*256.);
    }
    let mut user_data = unsafe{
        slice::from_raw_parts_mut(ran.as_mut_ptr() as *mut u8, shape.0*shape.1*shape.2*shape.3 * 4)
    };
    let listener = TcpListener::bind(server_address).unwrap();
    println!("addr: {}", server_address);
    let mut socket = TcpStream::connect(client_address).unwrap();
    println!("sending ");
    let mut data: Vec<u8> = vec![0; 30000];
    let mut buffer: &mut [u8] = data.as_mut_slice();
    let sy_time = SystemTime::now();
    socket.write(user_data);
    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        println!("server_session connect!");
        loop {
            if let Err(_) =
            stream.read(buffer)
            {
                continue;
            }
            println!("total measured time: {:?}", SystemTime::now().duration_since(sy_time).unwrap().as_micros());
        }
    }
 }

'''


def get_client_address(index, model_path):
    return r'''{
    "server_address": "127.0.0.1:'''+ str(32100+index) + r'''",
    "client_address": "127.0.0.1:'''+ str(32100) + r'''",
    "attestation_address": "127.0.0.1:'''+ str(4240+index) + r'''",
    "sp_address": "127.0.0.1:1310",
    "model": ["resnet18", "mobilenetv1"],
    "model_path": "'''+ str(model_path) + r'''",
    "input_size": [1,3,224,224]
}'''


if __name__ == "__main__":
    model_path = sys.argv[1] if sys.argv[1] else "/home/lww/project/02_Project/01-rust/cluster-inference/model/resnet50"
    num_models = len(os.listdir(model_path))
    target_dir = sys.argv[2] if sys.argv[2] else "/home/lww/project/02_Project/01-rust//"
    for i in range(num_models+1):
        if i == num_models:
            root = os.path.join(target_dir, 'client')
            cargo = os.path.join(root, '.cargo')
            src = os.path.join(root, 'src')
            m_path = os.path.join(model_path, str(0))
            print(m_path)
            my_mkdirs(root)
            my_mkdirs(cargo)
            my_mkdirs(src)
            with open(os.path.join(cargo, 'config'), 'w') as config, \
                open(os.path.join(src, 'main.rs'), 'w') as main, \
                open(os.path.join(root, 'build.rs'), 'w') as build, \
                open(os.path.join(root, 'Cargo.toml'), 'w') as toml, \
                open(os.path.join(root, 'Cargo.lock'), 'w') as lock, \
                open(os.path.join(root, 'config'), 'w') as address:
                config.write(get_config())
                main.write(get_client_main())
                build.write(get_build(m_path))
                toml.write(get_client_toml())
                lock.write(get_lock())
                address.write(get_client_address(i, m_path))
        else:
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
                open(os.path.join(root, 'Cargo.lock'), 'w') as lock, \
                open(os.path.join(root, 'config'), 'w') as address:
                config.write(get_config())
                main.write(get_main())
                build.write(get_build(m_path))
                toml.write(get_toml())
                lock.write(get_lock())
                address.write(get_address(i, m_path))
