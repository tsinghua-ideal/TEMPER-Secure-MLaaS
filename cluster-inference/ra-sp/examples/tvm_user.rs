extern crate clap;
use byteorder::{NetworkEndian, WriteBytesExt};
use ra_sp::{context::verify_quote, context::verify_quote_with_sigs, SpConfigs};
// use byteorder::{NetworkEndian, WriteBytesExt};
use clap::App;
use mbedtls::ssl::config::{Endpoint, Preset, Transport};
use mbedtls::ssl::{Config, Context};
use mbedtls::x509::Certificate;
use ra_common::tcp::tcp_connect;
use sgx_crypto::keys;
use sgx_crypto::random::Rng;
use std::io::{BufReader, Read, Write};
use std::sync::Arc;
use std::time::Duration;

fn parse_config_file(path: &str) -> SpConfigs {
    serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap()
}

fn main() {
    let matches = App::new("tvm-user")
        .version("1.0")
        .author("simplelin. ")
        .about("Do remote attestation")
        .args_from_usage(
            "-e   --enclave=[String] 'Sets IP and Port for sgx task enclave,such as: \"127.0.0.1:7777\"'
            -n   --number=[u8]     'Serial number for the enclave that will send quote to client,\"0-255\".such as:0' ")
        .get_matches();
    let enclave = matches.value_of("enclave").unwrap_or("127.0.0.1:7777");
    let number = matches.value_of("number").unwrap_or("0"); //.as_bytes();
    let number = number
        .parse::<usize>()
        .expect("the input value is invalid,that should be a id and type is \"u8\"");
    let timeout = Duration::from_secs(5);
    let mut sche_stream = tcp_connect(enclave, timeout).expect("Client: Enclave connection failed");
    eprintln!("Client: connected to enclave {:?}.", enclave);

    eprintln!("Starting verify_quote .");
    let mut configs = parse_config_file("examples/data/settings.json");
    sche_stream.write_all(&[2u8; 4]).unwrap();
    verify_quote(&mut configs, &mut sche_stream).unwrap();
    let verify_result = verify_quote_with_sigs(configs, &mut sche_stream).unwrap();
    sche_stream.write_all(&verify_result).unwrap();
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
    let mut ctx = Context::new(rc_config);
    ctx.establish(sche_stream, None)
        .expect("tls establish error");
    let mut user_session = BufReader::new(ctx);
    let user_data = vec![2u8; number * 3 * 224 * 224];

    println!("Send user data to scheduler!");
    user_session
        .get_mut()
        .write_u32::<NetworkEndian>(user_data.len() as u32)
        .unwrap();
    user_session.get_mut().write_all(&user_data).unwrap();
    println!("Wait read result from scheduler!");
    // let len = user_session.read_u32::<NetworkEndian>().unwrap() as usize;
    let len = user_data.len() / (3 * 224 * 224) * 1000;
    let mut resnet_result = vec![0u8; len];
    println!("Read the length of resnet: {:?}", &len);
    user_session.read_exact(&mut resnet_result).unwrap();
    println!("Read the length of resnet: {:?}", resnet_result.len());
    println!(
        "msg: {:?}",
        &resnet_result[resnet_result.len() - 16..resnet_result.len()]
    );
    user_session.get_mut().close();
}
