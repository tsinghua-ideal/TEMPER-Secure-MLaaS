use crate::SpRaResult;
use http::{HeaderMap, StatusCode};
use serde::{Deserialize, Serialize};
use std::net::IpAddr;

#[derive(Deserialize, Debug, Clone)]
pub struct AttestationInfo {
    pub id: u8,
    pub client_ip: String,
    pub enclave_port: String,
    pub sp_private_key_pem_path: String,
    pub sigstruct_path: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct SpConfig {
    pub enclave_id: u8,
    pub linkable: bool,
    pub random_nonce: bool,
    pub use_platform_service: bool,
    pub spid: String,
    pub primary_subscription_key: String,
    pub secondary_subscription_key: String,
    pub quote_trust_options: Vec<String>,
    pub pse_trust_options: Option<Vec<String>>,
    pub ias_root_cert_pem_path: String,
    pub sp_private_key_pem_path: String,
    pub sigstruct_path: String,
    pub enclave_port: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct SpConfigs {
    pub linkable: bool,
    pub random_nonce: bool,
    pub use_platform_service: bool,
    pub spid: String,
    pub primary_subscription_key: String,
    pub secondary_subscription_key: String,
    pub quote_trust_options: Vec<String>,
    pub pse_trust_options: Option<Vec<String>>,
    pub ias_root_cert_pem_path: String,
    pub listener_address: String,
    pub slaves: Vec<AttestationInfo>,
}

impl SpConfigs {
    pub fn generate_spconfig(&self, id: u8, peer_ip: &IpAddr) -> SpRaResult<SpConfig> {
        let mut sp_private_key_pem_path = "".to_string();
        let mut enclave_port = "".to_string();
        let mut exit: bool = false;
        let mut sigstruct_path = "".to_string();
        println!("slave id: {:?} ", id);
        //let id = idstr.parse::<u8>().expect("the input value is invalid,that should be a id and type is \"u8\"");
        for slave in self.slaves.iter() {
            //slaves.push(slave.clone());
            //if slave.client_port.parse::<SocketAddr>()  == Ok(*peer_socket){
            if slave.client_ip.parse::<IpAddr>() == Ok(*peer_ip)
                && "127.0.0.1".parse::<IpAddr>() != Ok(*peer_ip)
            {
                sp_private_key_pem_path = slave.sp_private_key_pem_path.clone();
                sigstruct_path = slave.sigstruct_path.clone();
                enclave_port = slave.enclave_port.clone();
                exit = true;
                break;
            } else if "127.0.0.1".parse::<IpAddr>() == Ok(*peer_ip) && id == slave.id {
                sp_private_key_pem_path = slave.sp_private_key_pem_path.clone();
                sigstruct_path = slave.sigstruct_path.clone();
                enclave_port = slave.enclave_port.clone();
                exit = true;
                break;
            }
        }
        match true == exit {
            true => Ok(SpConfig {
                sp_private_key_pem_path,
                sigstruct_path,
                enclave_port,
                enclave_id: id,
                linkable: self.linkable.clone(),
                random_nonce: self.random_nonce.clone(),
                use_platform_service: self.use_platform_service.clone(),
                spid: self.spid.clone(),
                primary_subscription_key: self.primary_subscription_key.clone(),
                secondary_subscription_key: self.secondary_subscription_key.clone(),
                quote_trust_options: self.quote_trust_options.clone(),
                pse_trust_options: self.pse_trust_options.clone(),
                ias_root_cert_pem_path: self.ias_root_cert_pem_path.clone(),
            }),
            false => Err(super::error::SpRaError::ClientConfigNotFound),
        }
    }

    // pub fn generate_spconfig(&self,peer_ip:& IpAddr) -> SpRaResult<SpConfig> {
    //     let mut sp_private_key_pem_path = "".to_string();
    //     let mut exit: bool = false;
    //     let mut sigstruct_path = "".to_string();
    //     for slave in self.slaves.iter() {
    //         //print!("{:#?} ", slave);
    //         //slaves.push(slave.clone());
    //         //if slave.client_port.parse::<SocketAddr>()  == Ok(*peer_socket){
    //         if slave.client_ip.parse::<IpAddr>()  ==  Ok(*peer_ip){
    //             sp_private_key_pem_path = slave.sp_private_key_pem_path.clone();
    //             sigstruct_path = slave.sigstruct_path.clone();
    //             exit = true;
    //             break;
    //         }
    //     }
}

//send http response with socket
#[derive(Serialize, Deserialize)]
pub struct HttpRespWrap {
    #[serde(with = "http_serde::header_map")]
    pub map: HeaderMap,
    #[serde(with = "http_serde::status_code")]
    pub statu: StatusCode,
}
