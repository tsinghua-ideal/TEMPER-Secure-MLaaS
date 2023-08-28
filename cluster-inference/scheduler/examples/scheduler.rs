use enclave_scheduler::handle_client;
use enclave_scheduler::task_schedule;
use enclave_scheduler::threadpool::ThreadPool;
use ra_enclave::context::EnclaveRaContext;
use ra_enclave::error::EnclaveRaError;
use ra_enclave::tls_enclave::attestation;
use std::sync::Mutex;
use std::{
    net::{TcpListener},
    sync::Arc,
};
use mbedtls::ssl::config::{Endpoint, Preset, Transport};
use mbedtls::ssl::Config;
use mbedtls::x509::Certificate;
use mbedtls::pk::Pk;
use sgx_crypto::random::Rng;
use sgx_crypto::keys;

fn read_config() ->(Option<String>, Option<String>){
    let config = include_str!(concat!(env!("PWD"), "/config"));
    let config: serde_json::Value = serde_json::from_str(config).unwrap();
    let client_address = config["attestation_address"].as_str().unwrap().to_string();
    let sch_addr = config["sch_server_address"].as_str().unwrap().to_string();
    // println!("{:#?} {:#?}", client_address, sp_address);
    (Some(client_address), Some(sch_addr))
}

fn do_multi_thread_task(enclave_racontext: Result<EnclaveRaContext, EnclaveRaError>, sch_addr: Option<String>){
    let mut config = Config::new(Endpoint::Server, Transport::Stream, Preset::Default);
    let rng = Rng::new();
    config.set_rng(Arc::new(rng.inner));
    let cert = Arc::new(Certificate::from_pem_multiple(keys::PEM_CERT).unwrap());
    let key = Arc::new(Pk::from_private_key(keys::PEM_KEY, None).unwrap());
    config.push_cert(cert, key).unwrap();
    let rc_config = Arc::new(config);
    
    eprintln!("Schedule: Listening at: {:#?}", sch_addr);
    let listener = TcpListener::bind(sch_addr.unwrap().as_str()).unwrap();
    // let mut thread_vec: Vec<thread::JoinHandle<()>> = Vec::new();
    let pool = ThreadPool::new(10);
    eprintln!("Schedule: ThreadPool::new !");
    {
        let config = Arc::clone(&rc_config);
        pool.execute(||{
            task_schedule(config);
        });
    }
    eprintln!("Schedule: wait for user or enclave !");
    let _own_racontext = Arc::new(Mutex::new(enclave_racontext.unwrap()));
    for stream in listener.incoming() {
        let config = Arc::clone(&rc_config);
        let stream = stream.unwrap();
        let racon = Arc::clone(&_own_racontext);
        pool.execute( || {
            handle_client(stream, racon, config);
        });
    }
}
fn main() -> std::io::Result<()> {
    println!("Attestation start ...");
    let (attest_addr, sch_addr) = read_config();
    // let mut enclave_racontext = attestation(attest_addr.unwrap().as_str());
    let mut enclave_racontext: Result<EnclaveRaContext, EnclaveRaError> =
        Err(EnclaveRaError::EnclaveNotTrusted);
    while enclave_racontext.is_err() {
        let addr = attest_addr.as_ref().unwrap();
        enclave_racontext = attestation(addr.as_str().as_ref());
    }
    // let mut client_stream = tcp_accept(bind_addr.unwrap().as_str()).expect("Enclave: init TCP server failed");
    println!("Attestation end!");
    do_multi_thread_task(enclave_racontext, sch_addr);

    // for handle in thread_vec {
    //     handle.join().unwrap();
    // }

    Ok(())
}
