/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

extern crate byteorder;
extern crate mbedtls;
extern crate ndarray;
extern crate rand;
extern crate tvm_graph_rt;

use byteorder::{NetworkEndian, WriteBytesExt};
use mbedtls::pk::Pk;
use mbedtls::ssl::config::{Endpoint, Preset, Transport};
use mbedtls::ssl::{Config, Context};
use mbedtls::x509::Certificate;
use ra_enclave::context::EnclaveRaContext;
use ra_enclave::tls_enclave::attestation_message;
use sgx_crypto::keys;
use sgx_crypto::random::Rng;
use std::convert::TryInto;
use std::net::TcpListener;

use ra_common::tcp::tcp_connect;
use serde_json::Value;
use std::time::Duration;

use std::{
    convert::TryFrom as _,
    io::{BufReader, Read, Write},
    sync::Arc,
    // thread,
    time::{SystemTime, UNIX_EPOCH},
};
//use ndarray::{Array, Array4};

fn timestamp() -> i64 {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let ms = since_the_epoch.as_secs() as i64 * 1000i64
        + (since_the_epoch.subsec_nanos() as f64 / 1_000_000.0) as i64;
    ms
}
fn read_config() -> (Option<String>, Option<String>) {
    let config = include_str!(concat!(env!("PWD"), "/config"));
    let config: serde_json::Value = serde_json::from_str(config).unwrap();
    let client_address = config["client_address"].as_str().unwrap().to_string();
    let sp_address = config["sp_address"].as_str().unwrap().to_string();
    // println!("{:#?} {:#?}", client_address, sp_address);
    (Some(client_address), Some(sp_address))
}
fn main() {
    println!("Attestation start ...");
    let (client_addr, sp_addr) = read_config();
    let mut enclave_racontext = attestation_message(
        client_addr.unwrap().as_str(),
        sp_addr.unwrap().as_str(),
        keep_message,
    )
    .unwrap();
    println!("Attestation end!");
    println!("do tvm start ...");
    // do_tvm(&mut enclave_racontext);
    execute_task(&mut enclave_racontext);
    println!("do tvm end!");
}

pub fn keep_message(session: BufReader<mbedtls::ssl::Context>) {
    let mut sess = session;
    let msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque non placerat risus, et lobortis quam. Mauris velit lorem, elementum id neque a, aliquet tempus turpis. Nam eu congue urna, in semper quam. Ut tristique gravida nunc nec feugiat. Proin tincidunt massa a arcu volutpat, sagittis dignissim velit convallis. Cras ac finibus lorem, nec congue felis. Pellentesque fermentum vitae ipsum sed gravida. Nulla consectetur sit amet erat a pellentesque. Donec non velit sem. Sed eu metus felis. Nullam efficitur consequat ante, ut commodo nisi pharetra consequat. Ut accumsan eget ligula laoreet dictum. Maecenas tristique porta convallis. Suspendisse tempor sodales velit, ac luctus urna varius eu. Ut ultrices urna vestibulum vestibulum euismod. Vivamus eu sapien urna.";
    sess.get_mut()
        .write_u32::<NetworkEndian>(msg.len() as u32)
        .unwrap();
    write!(&mut sess.get_mut(), "{}", msg).unwrap();
}
#[allow(dead_code)]
pub fn do_tvm(racontext: &mut EnclaveRaContext) {
    let config = include_str!(concat!(env!("PWD"), "/config"));
    let config: Value = serde_json::from_str(config).unwrap();
    let server_address = config["server_address"].as_str().unwrap();
    //let client_address = config[3];
    let syslib = tvm_graph_rt::SystemLibModule::default();
    let graph_json = include_str!(concat!(env!("OUT_DIR"), "/graph.json"));
    let params_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/params.bin"));
    let params = tvm_graph_rt::load_param_dict(params_bytes).unwrap();

    let graph = tvm_graph_rt::Graph::try_from(graph_json).unwrap();
    let mut exec = tvm_graph_rt::GraphExecutor::new(graph, &syslib).unwrap();
    exec.load_params(params);
    let listener = TcpListener::bind(server_address).unwrap();
    println!("TVM run: listening at {}", server_address);
    for stream in listener.incoming() {
        let mut stream = stream.unwrap();

        //to verify the quote and private key
        let signer_key = &mut racontext.signer_key;
        //read rand data from user
        let mut rand_data = vec![0u8; 64];
        let mut rng = Rng::new();
        stream.read_exact(&mut rand_data).unwrap();
        let sign_data = signer_key
            .ecdsa_sign(&rand_data, &mut rng)
            .expect("Error when signing the user rang data!");
        stream
            .write_u32::<NetworkEndian>(sign_data.len().try_into().unwrap())
            .unwrap();
        stream.write_all(racontext.quote.as_ref()).unwrap();
        stream.write_all(sign_data.as_ref()).unwrap();

        let mut verify_data = [0u8; 4];
        stream.read_exact(&mut verify_data).unwrap();
        if verify_data != [1u8; 4] {
            eprintln!("an error occured when user verify the quote and sign!");
            continue;
        }
        let mut verify_data = [0u8; 4];
        stream.read_exact(&mut verify_data).unwrap();
        continue;
        let mut config = Config::new(Endpoint::Server, Transport::Stream, Preset::Default);
        let rng = Rng::new();
        config.set_rng(Arc::new(rng.inner));

        let cert = Arc::new(Certificate::from_pem_multiple(keys::PEM_CERT).unwrap());
        let key = Arc::new(Pk::from_private_key(keys::PEM_KEY, None).unwrap());
        config.push_cert(cert, key).unwrap();

        let rc_config = Arc::new(config);
        let mut ctx = Context::new(rc_config);
        ctx.establish(stream, None).unwrap();
        let mut server_session = BufReader::new(ctx);
        println!("TVM run: a new user is coming ...");
        if let Err(_) = server_session.read(
            exec.get_input("input")
                .unwrap()
                .data()
                .view()
                .as_mut_slice(),
        ) {
            continue;
        }
        let ts1 = timestamp();
        println!("TimeStamp: {}", ts1);
        let sy_time = SystemTime::now();
        exec.run();
        let duration = SystemTime::now()
            .duration_since(sy_time)
            .unwrap()
            .as_micros();
        server_session
            .get_mut()
            .write(exec.get_output(0).unwrap().data().as_slice())
            .unwrap();
        println!(
            "output len: {:?}",
            exec.get_output(0).unwrap().data().as_slice().len()
        );
        println!("{:?}", duration);
        //only try once
        break;
    }
}

pub fn execute_task(racontext: &mut EnclaveRaContext) {
    let config = include_str!(concat!(env!("PWD"), "/config"));
    let config: Value = serde_json::from_str(config).unwrap();
    let schedule_address = config["schedule_address"].as_str().unwrap();
    //let client_address = config[3];
    let syslib = tvm_graph_rt::SystemLibModule::default();
    let graph_json = include_str!(concat!(env!("OUT_DIR"), "/graph.json"));
    let params_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/params.bin"));
    let params = tvm_graph_rt::load_param_dict(params_bytes).unwrap();

    let graph = tvm_graph_rt::Graph::try_from(graph_json).unwrap();
    let mut exec = tvm_graph_rt::GraphExecutor::new(graph, &syslib).unwrap();
    exec.load_params(params);
    // let listener = TcpListener::bind(server_address).unwrap();
    let timeout = Duration::from_secs(5);
    eprintln!("TVM run: connect to the schedule: {}", schedule_address);
    let mut schedule_stream =
        tcp_connect(&schedule_address, timeout).expect("Client: Enclave connection failed");

    // send [1u8;4] to show ,this is a enclave service
    schedule_stream.write_all(&[1u8; 4]).unwrap();
    schedule_stream.write_all(racontext.quote.as_ref()).unwrap();
    //read rand data from user
    let mut rand_data = vec![0u8; 64];
    let mut rng = Rng::new();
    schedule_stream.read_exact(&mut rand_data).unwrap();

    let signer_key = &mut racontext.signer_key;
    let sign_data = signer_key
        .ecdsa_sign(&rand_data, &mut rng)
        .expect("Error when signing the user rang data!");
    schedule_stream
        .write_u32::<NetworkEndian>(sign_data.len().try_into().unwrap())
        .unwrap();

    schedule_stream.write_all(sign_data.as_ref()).unwrap();
    let mut verify_result = vec![0u8; 4];
    schedule_stream.read_exact(&mut verify_result).unwrap();
    if verify_result == [1u8; 4] {
        eprintln!("Schedule verify own successfully!");
    }
    //new a TLS context
    let cert = Arc::new(Certificate::from_pem_multiple(keys::PEM_CERT).unwrap());
    let mut config = Config::new(Endpoint::Client, Transport::Stream, Preset::Default);
    let rng = Rng::new();
    config.set_rng(Arc::new(rng.inner));
    config.set_ca_list(cert, None);

    // let cert = Arc::new(Certificate::from_pem_multiple(keys::PEM_CERT).unwrap());
    // let key = Arc::new(Pk::from_private_key(keys::PEM_KEY, None).unwrap());
    // config.push_cert(cert, key).unwrap();

    let rc_config = Arc::new(config);

    loop {
        println!("New a TLS connection ...");
        let mut ctx = Context::new(rc_config.clone());
        let mut io = schedule_stream
            .try_clone()
            .expect("try to clone a stream failed!");

        let mut msg = vec![0u8; 23];
        let mut loop_num = 1u8;
        while msg[0..8] != [8u8; 8] {
            io.read(&mut msg).unwrap();
            println!(
                "msg {:#?}",
                msg.clone()
                    .into_iter()
                    .map(|x| format!(" {:x}", x))
                    .collect::<String>()
            );
            if msg[0..8] == [8u8; 8] {
                loop_num = msg[8];
            }
        }
        io.write(&msg[0..8]).unwrap();
        println!("Try to establish a tls to schedule ...");
        ctx.establish(io, None).expect("tls establish error");
        let mut schedule_session = BufReader::new(ctx);
        println!("TVM run: a new user is coming ...");
        println!(
            "Need input data len: {:?}",
            exec.get_input("data")
                .unwrap()
                .data()
                .view()
                .as_mut_slice()
                .len()
        );
        while loop_num != 0 {
            if let Err(_) = schedule_session
                .read_exact(exec.get_input("data").unwrap().data().view().as_mut_slice())
            {
                continue;
            }
            let ts1 = timestamp();
            println!("TimeStamp: {}", ts1);
            let sy_time = SystemTime::now();
            exec.run();
            let duration = SystemTime::now()
                .duration_since(sy_time)
                .unwrap()
                .as_micros();
            let result = exec.get_output(0).unwrap().data().as_slice();
            println!("output len: {:?}  {:?}", result.len(), result.len());
            println!("msg: {:?}", &result[result.len() - 16..result.len()]);
            schedule_session
                .get_mut()
                .write_u32::<NetworkEndian>(result.len() as u32)
                .unwrap();
            schedule_session
                .get_mut()
                .write_all(result)
                .expect("write the result to schedule!");

            println!("exec time is {:?} ms, ", duration,);
            loop_num -= 1;
        }
        schedule_session.get_mut().close();
    }
}
