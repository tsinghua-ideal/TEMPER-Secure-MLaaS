use crate::error::EnclaveRaError;
use crate::local_attestation;
use crate::EnclaveRaResult;
use ra_common::derive_secret_keys;
use ra_common::msg::{Quote, RaMsg2, RaMsg3, RaMsg4};
use sgx_crypto::cmac::{Cmac, MacTag};
use sgx_crypto::digest::sha256;
use sgx_crypto::key_exchange::OneWayAuthenticatedDHKE;
use sgx_crypto::random::Rng;
use sgx_crypto::signature::{VerificationKey, SigningKey};
use sgx_isa::{Report, Targetinfo};
use std::io::{Read, Write};
use std::mem::size_of;
use std::time::SystemTime;

pub struct EnclaveRaContext {
    pub key_exchange: Option<OneWayAuthenticatedDHKE>,
    pub sp_vkey: VerificationKey,
    pub signer_key: SigningKey,
    pub quote: Quote
}

impl EnclaveRaContext {
    pub fn init(sp_vkey_pem: &str) -> EnclaveRaResult<Self> {
        let mut rng = Rng::new();
        let mut signer_rng = Rng::new();
        let key_exchange = OneWayAuthenticatedDHKE::generate_keypair(&mut rng)?;
        let signer_key = SigningKey::generate_keypair(&mut signer_rng).expect("generate signing key pair failed!");
        Ok(Self {
            sp_vkey: VerificationKey::new(sp_vkey_pem.as_bytes())?,
            key_exchange: Some(key_exchange),
            signer_key,
            quote: [0u8;size_of::<Quote>()]
        })
    }

    pub fn do_attestation(
        &mut self,
        mut client_stream: &mut (impl Read + Write),
    ) -> EnclaveRaResult<(MacTag, MacTag)> {
        let (sk, mk) = self.process_msg_2(client_stream).unwrap();
        let msg4: RaMsg4 = bincode::deserialize_from(&mut client_stream).unwrap();
        if !msg4.is_enclave_trusted {
            return Err(EnclaveRaError::EnclaveNotTrusted);
        }
        match msg4.is_pse_manifest_trusted {
            Some(t) => {
                if !t {
                    return Err(EnclaveRaError::PseNotTrusted);
                }
            }
            None => {}
        }
        Ok((sk, mk))
    }

    // Return (signing key, master key)
    pub fn process_msg_2(
        &mut self,
        mut client_stream: &mut (impl Read + Write),
    ) -> EnclaveRaResult<(MacTag, MacTag)> {

        //generate the msg_1, send to sp enclave
        let g_a = self.key_exchange.as_ref().unwrap().get_public_key()?;
        bincode::serialize_into(&mut client_stream, &g_a).unwrap();

        let msg2: RaMsg2 = bincode::deserialize_from(&mut client_stream).unwrap();

        // Verify and derive KDK and then other secret keys
        let mut rng = Rng::new();
        let kdk = self
            .key_exchange
            .take()
            .unwrap()
            .verify_and_derive(&msg2.g_b, &msg2.sign_gb_ga, &mut self.sp_vkey, &mut rng)
            .unwrap();
        let mut kdk_cmac = Cmac::new(&kdk)?;
        let (smk, sk, mk, vk) = derive_secret_keys(&mut kdk_cmac)?;
        let mut smk = Cmac::new(&smk)?;

        // Verify MAC tag of MSG2
        msg2.verify_mac(&mut smk)?;

        // Obtain SHA-256(g_a || g_b || vk)
        let mut verification_msg = Vec::new();
        verification_msg.write_all(g_a.as_ref()).unwrap();
        verification_msg.write_all(&msg2.g_b).unwrap();
        verification_msg.write_all(&vk).unwrap();
        let verification_digest = sha256(&verification_msg[..])?;
        //add a ecdsa public key to report data for a sign
        let sy_time = SystemTime::now();
        let signer_public_key = self.signer_key.get_public_key()?;
        println!("public_key generation time: {:?}", SystemTime::now().duration_since(sy_time).unwrap().as_micros());
        let mut _report_data = [0u8; 64];
        (&mut _report_data[..(verification_digest.len())]).clone_from_slice(&verification_digest);
        
        //signer_public_key.len() ==33, so we need to copy to _report_data at (verification_digest.len()-1)
        (&mut _report_data[(verification_digest.len()-1)..]).clone_from_slice(&signer_public_key);
        // Obtain Quote
        let sy_time = SystemTime::now();
        self.get_quote(&_report_data[..], client_stream)?;
        println!("quote generation time: {:?}", SystemTime::now().duration_since(sy_time).unwrap().as_micros());
        // Send MAC for msg3 to client
        let msg3 = RaMsg3::new(&mut smk, g_a, None, self.quote)?;
        client_stream.write_all(&msg3.mac).unwrap();

        Ok((sk, mk))
    }

    /// Get quote from Quote Enclave. The length of report_data must be <= 64 bytes.
    pub fn get_quote(
        &mut self,
        report_data: &[u8],
        client_stream: &mut (impl Read + Write),
    ) -> EnclaveRaResult<Quote> {
        if report_data.len() > 64 {
            return Err(EnclaveRaError::ReportDataLongerThan64Bytes);
        }

        // Obtain QE's target info to build a report for local attestation.
        // Then, send the report back to client.
        let mut _report_data = [0u8; 64];
        (&mut _report_data[..(report_data.len())]).copy_from_slice(report_data);
        let mut target_info = [0u8; Targetinfo::UNPADDED_SIZE];
        client_stream.read_exact(&mut target_info).unwrap();
        let target_info = Targetinfo::try_copy_from(&target_info).unwrap();
        let report = Report::for_target(&target_info, &_report_data);


        //add some custom data to report, then get the quote
        client_stream.write_all(report.as_ref()).unwrap();

        // Obtain quote and QE report from client
        // let mut quote = [0u8; size_of::<Quote>()];
        client_stream.read_exact(&mut self.quote[..]).unwrap();
        let qe_report_len = 432usize;
        let mut qe_report = vec![0u8; qe_report_len];
        client_stream.read_exact(&mut qe_report[..]).unwrap();

        // Verify that the report is generated by QE
        local_attestation::verify_local_attest(&qe_report[..])
            .map_err(|e| EnclaveRaError::LocalAttestation(e))?;
        Ok(self.quote)
    }
}
