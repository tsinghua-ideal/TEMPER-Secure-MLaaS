extern crate clap;

use byteorder::{NetworkEndian, WriteBytesExt}; 
use attest_client::ClientRaContext;
use ra_common::tcp::tcp_connect;
use std::time::Duration;
use std::io::Write;
use clap::App;

fn main() {
    let matches = App::new("attest-client")
                .version("1.0")
                .author("simplelin. ")
                .about("Do remote attestation")
                .args_from_usage(
                    "-e   --enclave=[String] 'Sets IP and Port for enclave,such as: \"127.0.0.1:7777\"'
                    -s   --server=[String] 'Sets IP and Port for service provide,such as: \"192.168.1.1:1234\"'
                    -n   --number=[u8]     'Serial number for the enclave that will send quote to client,\"0-255\".such as:0' ")
                .get_matches();
    let enclave  = matches.value_of("enclave").unwrap_or("127.0.0.1:7777");
    let service  = matches.value_of("server").unwrap_or("127.0.0.1:1234");
    let number  = matches.value_of("number").unwrap_or("0");//.as_bytes();
    let number  = number.parse::<u8>().expect("the input value is invalid,that should be a id and type is \"u8\"");
    let timeout = Duration::from_secs(5);
    let mut enclave_stream =
        tcp_connect(enclave, timeout).expect("Client: Enclave connection failed");
    eprintln!("Client: connected to enclave {:?}.", enclave);

    let mut sp_stream =
        tcp_connect(service, timeout).expect("Client: SP connection failed");
    eprintln!("Client: connected to SP {:?}.", service);
    sp_stream.write_u8(number).unwrap();
    let context = ClientRaContext::init().unwrap();
    context
        .do_attestation(&mut enclave_stream, &mut sp_stream)
        .unwrap();
    let msg = "127.0.0.1:3333";
    sp_stream.write_u32::<NetworkEndian>(msg.len() as u32).unwrap();
    //write!(&mut sp_stream, "{}", msg).unwrap();
    sp_stream.write(msg.as_bytes()).unwrap();
    eprintln!("Client: done!");
}
