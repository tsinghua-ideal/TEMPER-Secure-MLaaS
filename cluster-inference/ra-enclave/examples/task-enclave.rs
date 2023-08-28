mod sp_vkey;

use crate::sp_vkey::SP_VKEY_PEM;
use byteorder::{NetworkEndian};
use ra_common::tcp::tcp_accept;
use ra_enclave::EnclaveRaContext;
use sgx_crypto::tls_psk::server;
use sgx_crypto::random::Rng;

fn main() {
    let client_addr = "localhost:7777";
    let mut client_stream = tcp_accept(&client_addr).expect("Enclave: Client connection failed");
    eprintln!("Enclave: connected to client.{:#?}", &client_addr);
    let context = EnclaveRaContext::init(SP_VKEY_PEM).unwrap();
    let (_signing_key, master_key) = context.do_attestation(&mut client_stream).unwrap();

    // talk to SP directly from now on
    let sp_addr = "localhost:1235";
    let mut sp_stream = tcp_accept(sp_addr).expect("Enclave: SP connection failed!");

    // establish TLS-PSK with SP; enclave is the server
    let mut psk_callback = server::callback(&master_key);
    let mut rng = Rng::new();
    let config = server::config(&mut rng, &mut psk_callback);
    let mut ctx = server::context(&config);

    // begin secure communication
    let mut session = ctx.establish(&mut sp_stream, None).unwrap();
    let msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque non placerat risus, et lobortis quam. Mauris velit lorem, elementum id neque a, aliquet tempus turpis. Nam eu congue urna, in semper quam. Ut tristique gravida nunc nec feugiat. Proin tincidunt massa a arcu volutpat, sagittis dignissim velit convallis. Cras ac finibus lorem, nec congue felis. Pellentesque fermentum vitae ipsum sed gravida. Nulla consectetur sit amet erat a pellentesque. Donec non velit sem. Sed eu metus felis. Nullam efficitur consequat ante, ut commodo nisi pharetra consequat. Ut accumsan eget ligula laoreet dictum. Maecenas tristique porta convallis. Suspendisse tempor sodales velit, ac luctus urna varius eu. Ut ultrices urna vestibulum vestibulum euismod. Vivamus eu sapien urna.";
    session
        .write_u32::<NetworkEndian>(msg.len() as u32)
        .unwrap();
    write!(&mut session, "{}", msg).unwrap();
    eprintln!("Enclave: done!");
}
