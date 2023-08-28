use byteorder::{NetworkEndian, ReadBytesExt, WriteBytesExt};
use ra_common::msg::Quote;
use ra_enclave::context::EnclaveRaContext;
use ra_enclave::error::EnclaveRaError;
use sgx_crypto::random::Rng;
use sgx_crypto::signature::VerificationKey;
// use std::fmt::Result;
use mbedtls::rng::Random;
use mbedtls::ssl::{Config, Context};
use std::io::ErrorKind;
use std::mem::size_of;
use std::sync::Mutex;
use std::{
    io::{BufReader, Error, Read, Write},
    net::{Shutdown, TcpStream},
    sync::Arc,
    thread, time,
    time::{SystemTime, UNIX_EPOCH},
    option::Option
};

use crate::tvm_task_schedule::USER_DATAS;
use crate::tvm_task_schedule::RESULTS;
use crate::tvm_task_schedule::TENCLAVES;
#[allow(dead_code)]
enum Classific {
    UserLabel,
    EnclaveLabel,
}
#[allow(dead_code)]
pub struct Task {
    cla: Classific,
    id: u8,
}

pub struct TaskEnclave {
    // mrenclave = quote[112..144];
    // mrsigner  = quote[176..208];
    mrenclave: Vec<u8>,
    rand_data: Vec<u8>,
    rand_sig: Vec<u8>,
    // let ec_public_key = &mut quote[399..432];
    ec_public_key: Vec<u8>,
    statu: u8, // 0 not trust; 1 trust
    next_socket_addr: String,
    head_socket_addr: String,
    quote: [u8; size_of::<Quote>()],
    stream: TcpStream,
}

impl TaskEnclave {
    pub fn new(quote: Quote, stream: TcpStream) -> TaskEnclave {
        TaskEnclave {
            mrenclave: quote[112..144].to_owned(),
            rand_data:Vec::with_capacity(64),
            rand_sig:Vec::new(),
            ec_public_key: quote[399..432].to_owned(),
            statu: 0,
            next_socket_addr: "".to_string(),
            head_socket_addr: "".to_string(),
            quote,
            stream,
        }
    }
    fn verify_sig(&mut self) -> Result<(), EnclaveRaError> {
        let mut rng = Rng::new();
        self.rand_data.resize(64, 0);
        rng.inner
            .random(&mut self.rand_data)
            .expect("error for rand data!");
        self.stream.write_all(&self.rand_data).unwrap();
        let sig_len = self.stream.read_u32::<NetworkEndian>().unwrap() as usize;
        self.rand_sig.resize(sig_len, 0);
        self.stream.read_exact(&mut self.rand_sig[..]).unwrap();
        let mut verify_key = VerificationKey::new_from_binary(&self.ec_public_key)
            // let mut verify_key = VerificationKey::new(&ec_public_key[..33])
            .expect("get new verify public key failed!");
        verify_key
            .verify(&self.rand_data, &self.rand_sig)
            .expect("verify failed!");
        self.stream.write_all(&[1u8;4]).unwrap();
        Ok(())
    }
    pub fn get_stream(&self) ->&  TcpStream{
        & self.stream
    }
}

pub struct User {
    data: Vec<u8>,
    result: Vec<u8>,
    label: String,
    output_len: usize,
    stream: TcpStream,
    tls_session: Option<BufReader<Context>> ,
}

impl User {
    pub fn new(stream: TcpStream) -> User {
        let label = stream.peer_addr().unwrap().to_string();
        println!("user label: {:?}", label);
        User { data:Vec::new(),result:Vec::new(), label,output_len:0, stream,  tls_session: None}
    }
    fn for_attestation(
        &mut self,
        racon: Arc<Mutex<EnclaveRaContext>>,
    ) -> Result<(), EnclaveRaError> {
        //read rand data from user
        let mut rand_data = vec![0u8; 64];
        let mut rng = Rng::new();
        self.stream.read_exact(&mut rand_data).unwrap();
        self.stream
            .write_all(racon.lock().unwrap().quote.as_ref())
            .unwrap();
        let sign_data = racon
            .lock()
            .unwrap()
            .signer_key
            .ecdsa_sign(&rand_data, &mut rng)
            .expect("Error when signing the user rang data!");
        self.stream
            .write_u32::<NetworkEndian>(sign_data.len() as u32)
            .expect("write sign len error!");        
        self.stream.write_all(sign_data.as_ref()).unwrap();
        let mut verify_data = vec![0u8; 4];
        self.stream.read_exact(&mut verify_data).unwrap();
        match verify_data {
            x if x == vec![1u8; 4] => Ok(()),
            _ => Err(EnclaveRaError::EnclaveNotTrusted),
        }
    }

    fn send_enclave_quote(&mut self) -> Result<bool, EnclaveRaError> {
        let ens = TENCLAVES.read().unwrap();
        let ens_len = ens.len();
        self.stream
            .write_u32::<NetworkEndian>(ens_len as u32)
            .unwrap();
        for enclave in &*ens {
            self.stream.write_all(&enclave.rand_data).unwrap();
            self.stream.write_u32::<NetworkEndian>(enclave.rand_sig.len() as u32).unwrap();
            self.stream.write_all(&enclave.rand_sig).unwrap();
            self.stream.write_all(&enclave.quote).unwrap();
        }
        let mut verify_result = vec![0u8; ens_len];
        self.stream.read_exact(&mut verify_result).unwrap();
        match verify_result {
            x if x == vec![5u8; ens_len] => Ok(true),
            _ => Err(EnclaveRaError::EnclaveNotTrusted),
        }
    }

    fn read_user_data(&mut self, rc_config:Arc<Config>) -> Result<(), Error> {
        // self.stream.read_exact(&mut self.data).unwrap();
        let mut ctx = Context::new(rc_config.clone());
        let io = self.stream.try_clone().expect("try to clone a stream failed!");
        println!("Try to establish tls with a user");
        ctx.establish( io, None).expect("tls establish error");
        self.tls_session = Some(BufReader::new(ctx));
        let mut data = Vec::new();
        match self.tls_session.as_mut() {
            Some(s) => {
                let data_len = s.read_u32::<NetworkEndian>().unwrap() as usize;
                data.resize(data_len, 0);
                s.read_exact(&mut data).unwrap();
                self.output_len = data.len()/(3*224*224)*1000;
                let mut udatas =USER_DATAS.lock().unwrap();
                (*udatas).insert(self.label.clone(), data);
                Ok({})
            },
            None => {println!("can not read data from user tls!"); 
            Err(Error::new(ErrorKind::InvalidData, "can not read data from user tls!")) },
        }
    }

    fn send_result(&mut self)-> Result<(), Error>{        
        while self.output_len>0 { 
            let mut resultsmap=RESULTS.write().unwrap();           
            if resultsmap.len()> 0 && resultsmap.contains_key(&self.label){
                match self.tls_session.as_mut() {
                    Some(s) => {
                        self.result = (*resultsmap).remove(&self.label).unwrap();
                        // s.get_mut().write_u32::<NetworkEndian>(self.result.len() as u32).unwrap();
                        println!("Len of output to user({:?}) {:?}", self.label, self.result.len());
                        self.output_len-=self.result.len();
                        s.get_mut().write_all(&mut self.result).unwrap();                        
                        {}
                    }
                    None => {},
                }               
            }
            let ten_millis = time::Duration::from_millis(10000);
            thread::sleep(ten_millis);
        }        
        Ok(())
    }

    pub fn get_data(& self) -> & Vec<u8>{
        & self.data
    }
    pub fn get_stream(& self) -> & TcpStream{
        & self.stream
    }
    pub fn drop(self){
        drop(self.data);
        drop(self.result);
        drop(self.label);
        drop(self.tls_session.unwrap());
        drop(self.stream);
    }
}

pub fn handle_client(mut stream: TcpStream, racon: Arc<Mutex<EnclaveRaContext>>, rc_config:Arc<Config>) {
    let mut buffer = [0; 4];
    stream.read(&mut buffer).unwrap();
    if buffer[0..4] == [1; 4] {
        // enclave task        
        let mut quote = [0u8; size_of::<Quote>()];
        stream.read_exact(&mut quote[..]).unwrap();
        let mut taskencl = TaskEnclave::new(quote, stream);

        let verify_re = taskencl.verify_sig();
        eprintln!("a new enclave verify successfully!");
        match verify_re {
            Ok(_v) => TENCLAVES.clone().write().unwrap().push(taskencl),
            _ => {
                taskencl
                    .stream
                    .shutdown(Shutdown::Both)
                    .expect("shutdown call failed");
                eprint!("Verify the signature failed!");
            }
        }
        // TENCLAVES.clone().write().unwrap().push(taskencl);
    } else if buffer[0..4] == [2; 4] {
        // do user task
        let mut user = User::new(stream);
        user.for_attestation(racon)
            .expect("verify the schedule's quote");
        user.send_enclave_quote()
            .expect("send quotes to user, to do a verification!");
        println!("A new user verify successfully!");
        //read user's data, and then send to enclave for task; data must encrypto
        user.read_user_data(rc_config).expect("read data from user socket"); 

        println!("wait for send result to user!");
        user.send_result().expect("send comulation result to user!");
    }
}
