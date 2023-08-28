use crate::error::EnclaveRaError;
use crate::context::EnclaveRaContext;
use sgx_crypto::random::Rng;
use sgx_crypto::tls_psk::server;
use sgx_crypto::signature::SigningKey;
use ra_common::tcp::tcp_accept;
use std::net::{TcpStream};
use std::collections::HashMap;
use std::io::BufReader;
use serde::{Deserialize, Serialize};
use http::{HeaderMap, StatusCode};
use std::sync::Arc;
use mbedtls::ssl::Context;

pub const SP_VKEY_PEM: &str = "\
-----BEGIN RSA PUBLIC KEY-----\n
MIIBCgKCAQEAvtc94gzwX0KeL1HJVh6XdHPXXA4PYE+ClqWUvxp5ts1/nLQzJVcy\
1SHMGaPUCr+IZJBeWapkFpgnJnw7YzdQ2kA8k6GiN/k8hlQMWXA2nE0LDeOHX8i7\
fc31lWy5nHdAXj7SfC/YV5RC/yhkJ2cYNMB15VPRHGQRukdVmvHUFunxwfkHq5mM\
xWWAWO5Km490NCWP7CqBH6ezGm5jUhzYT/n5y5EaVpqwKVE1uYA//L4dFSE7aDzD\
CDb50B9uqPaEyKHwc2taLiSPvQjDQE3BpKTDOqsVnojd9br1vYW/uemYnnlOJbSr\
L7pYuPODmV02by5r+7hgXFQkTADwFQBCmwIDAQAB\n\
-----END RSA PUBLIC KEY-----\
\0";

#[derive(Serialize, Deserialize, Debug)]
pub struct HttpRespWrap{
    #[serde(with = "http_serde::header_map")] 
    pub map: HeaderMap,
    #[serde(with = "http_serde::status_code")] 
    pub statu: StatusCode,
}
pub fn attestation(bind_addr:&str)->Result<EnclaveRaContext, EnclaveRaError>{
    eprintln!("Enclave: listening at:{:?}", bind_addr);
    let mut enclave_stream = tcp_accept(bind_addr).expect("Enclave: Client connection failed");
    let mut encontext = EnclaveRaContext::init(SP_VKEY_PEM).unwrap();
    let (_signing_key, _master_key) = encontext.do_attestation(&mut enclave_stream).unwrap();
    Ok(encontext)
}    
pub fn attestation_message(bind_addr:&str, sp:&str, keep_message:fn(BufReader<Context>))->Result<EnclaveRaContext, EnclaveRaError>{
    eprintln!("Enclave: listening at:{:?}", bind_addr);
    let mut enclave_stream = tcp_accept(bind_addr).expect("Enclave: Client connection failed");
    let mut encontext = EnclaveRaContext::init(SP_VKEY_PEM).unwrap();
    let (_signing_key, master_key) = encontext.do_attestation(&mut enclave_stream).unwrap();

    // talk to SP directly from now on
    //let sp_port = 1235;
    let sp_stream = tcp_accept(sp).expect("Enclave: SP connection failed");
    // establish TLS-PSK with SP; enclave is the server
    let mut psk_callback = server::callback(&master_key);
    let rng = Rng::new();
    let config = server::config(rng, &mut psk_callback);
    let mut ctx = server::context(Arc::new(config));
    // begin secure communication
    ctx.establish(sp_stream, None).unwrap();
    eprintln!("Enclave: done!");
    let server_session = BufReader::new(ctx);
    keep_message(server_session);
    Ok(encontext)
}
pub fn attestation_get_report(client:&str, sp:&str, keep_message:fn(TcpStream, &mut HashMap<u8, (Vec<u8>, Vec<u8>)>), report: &mut HashMap<u8, (Vec<u8>, Vec<u8>)> )->Result<SigningKey, EnclaveRaError>{
    let mut client_stream = tcp_accept(client).expect("Enclave: Client connection failed");
    eprintln!("Enclave: connected to client.");
    let mut encontext = EnclaveRaContext::init(SP_VKEY_PEM).unwrap();
    let (_signing_key, _master_key) = encontext.do_attestation(&mut client_stream).unwrap();

    // talk to SP directly from now on
    //let sp_port = 1235;
    println!("wait for connect:");
    let sp_stream = tcp_accept(sp).expect("Enclave: SP connection failed");
    println!("wait for connect: done!");
    keep_message(sp_stream, report);
    Ok(encontext.signer_key)
}