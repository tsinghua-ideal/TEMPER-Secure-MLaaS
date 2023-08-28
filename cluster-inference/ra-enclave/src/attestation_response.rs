use crate::error::AttestationError;
use http::{HeaderMap, HeaderValue};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use sgx_crypto::certificate::X509Cert;

pub const CA_CERT_PEM: &str = "\
-----BEGIN CERTIFICATE-----\n\
MIIFSzCCA7OgAwIBAgIJANEHdl0yo7CUMA0GCSqGSIb3DQEBCwUAMH4xCzAJBgNV\
BAYTAlVTMQswCQYDVQQIDAJDQTEUMBIGA1UEBwwLU2FudGEgQ2xhcmExGjAYBgNV\
BAoMEUludGVsIENvcnBvcmF0aW9uMTAwLgYDVQQDDCdJbnRlbCBTR1ggQXR0ZXN0\
YXRpb24gUmVwb3J0IFNpZ25pbmcgQ0EwIBcNMTYxMTE0MTUzNzMxWhgPMjA0OTEy\
MzEyMzU5NTlaMH4xCzAJBgNVBAYTAlVTMQswCQYDVQQIDAJDQTEUMBIGA1UEBwwL\
U2FudGEgQ2xhcmExGjAYBgNVBAoMEUludGVsIENvcnBvcmF0aW9uMTAwLgYDVQQD\
DCdJbnRlbCBTR1ggQXR0ZXN0YXRpb24gUmVwb3J0IFNpZ25pbmcgQ0EwggGiMA0G\
CSqGSIb3DQEBAQUAA4IBjwAwggGKAoIBgQCfPGR+tXc8u1EtJzLA10Feu1Wg+p7e\
LmSRmeaCHbkQ1TF3Nwl3RmpqXkeGzNLd69QUnWovYyVSndEMyYc3sHecGgfinEeh\
rgBJSEdsSJ9FpaFdesjsxqzGRa20PYdnnfWcCTvFoulpbFR4VBuXnnVLVzkUvlXT\
L/TAnd8nIZk0zZkFJ7P5LtePvykkar7LcSQO85wtcQe0R1Raf/sQ6wYKaKmFgCGe\
NpEJUmg4ktal4qgIAxk+QHUxQE42sxViN5mqglB0QJdUot/o9a/V/mMeH8KvOAiQ\
byinkNndn+Bgk5sSV5DFgF0DffVqmVMblt5p3jPtImzBIH0QQrXJq39AT8cRwP5H\
afuVeLHcDsRp6hol4P+ZFIhu8mmbI1u0hH3W/0C2BuYXB5PC+5izFFh/nP0lc2Lf\
6rELO9LZdnOhpL1ExFOq9H/B8tPQ84T3Sgb4nAifDabNt/zu6MmCGo5U8lwEFtGM\
RoOaX4AS+909x00lYnmtwsDVWv9vBiJCXRsCAwEAAaOByTCBxjBgBgNVHR8EWTBX\
MFWgU6BRhk9odHRwOi8vdHJ1c3RlZHNlcnZpY2VzLmludGVsLmNvbS9jb250ZW50\
L0NSTC9TR1gvQXR0ZXN0YXRpb25SZXBvcnRTaWduaW5nQ0EuY3JsMB0GA1UdDgQW\
BBR4Q3t2pn680K9+QjfrNXw7hwFRPDAfBgNVHSMEGDAWgBR4Q3t2pn680K9+Qjfr\
NXw7hwFRPDAOBgNVHQ8BAf8EBAMCAQYwEgYDVR0TAQH/BAgwBgEB/wIBADANBgkq\
hkiG9w0BAQsFAAOCAYEAeF8tYMXICvQqeXYQITkV2oLJsp6J4JAqJabHWxYJHGir\
IEqucRiJSSx+HjIJEUVaj8E0QjEud6Y5lNmXlcjqRXaCPOqK0eGRz6hi+ripMtPZ\
sFNaBwLQVV905SDjAzDzNIDnrcnXyB4gcDFCvwDFKKgLRjOB/WAqgscDUoGq5ZVi\
zLUzTqiQPmULAQaB9c6Oti6snEFJiCQ67JLyW/E83/frzCmO5Ru6WjU4tmsmy8Ra\
Ud4APK0wZTGtfPXU7w+IBdG5Ez0kE1qzxGQaL4gINJ1zMyleDnbuS8UicjJijvqA\
152Sq049ESDz+1rRGc2NVEqh1KaGXmtXvqxXcTB+Ljy5Bw2ke0v8iGngFBPqCTVB\
3op5KBG3RjbF6RRSzwzuWfL7QErNC8WEy5yDVARzTA5+xmBc388v9Dm21HGfcC8O\
DD+gT9sSpssq0ascmvH49MOgjt1yoysLtdCtJW/9FZpoOypaHx0R+mJTLwPXVMrv\
DaVzWh5aiEx+idkSGMnX\n\
-----END CERTIFICATE-----\
\0";

#[derive(Deserialize, Debug)]
pub struct AttestationResponse {
    // header
    pub advisory_url: Option<String>,
    pub advisory_ids: Option<String>,
    pub request_id: String,
    // body
    pub id: String,
    pub timestamp: String,
    pub version: u16,
    pub isv_enclave_quote_status: String,
    pub isv_enclave_quote_body: String,
    pub revocation_reason: Option<String>,
    pub pse_manifest_status: Option<String>,
    pub pse_manifest_hash: Option<String>,
    pub platform_info_blob: Option<String>,
    pub nonce: Option<String>,
    pub epid_pseudonym: Option<String>,
}

impl AttestationResponse {
    pub fn from_response(
        headers: &HeaderMap,
        body: Vec<u8>,
    ) -> Result<Self, AttestationError> {
        let root_ca_cert = X509Cert::new_from_pem(CA_CERT_PEM.as_bytes()).unwrap();
        Self::verify_response(&root_ca_cert, &headers, &body[..])?;

        let body: Value = {
            let body = String::from_utf8(body).unwrap();
            serde_json::from_str(&body).unwrap()
        };
        // if cfg!(feature = "verbose") {
        //     eprintln!("==============headers Result==============");
        //     eprintln!("{:#?}", headers);
        //     eprintln!("==============================================");
        //     eprintln!("==============body Result==============");
        //     eprintln!("{:#?}", body);
        //     eprintln!("==============================================");
        // }
        let h = |x: &HeaderValue| x.to_str().unwrap().to_owned();
        let b = |x: &str| x.to_owned();
        Ok(Self {
            // header
            advisory_ids: headers.get("advisory-ids").map(h),
            advisory_url: headers.get("advisory-url").map(h),
            request_id: headers.get("request-id").map(h).unwrap(),
            // body
            id: body["id"].as_str().unwrap().to_owned(),
            timestamp: body["timestamp"].as_str().unwrap().to_owned(),
            version: body["version"].as_u64().unwrap() as u16,
            isv_enclave_quote_status: body["isvEnclaveQuoteStatus"].as_str().unwrap().to_owned(),
            isv_enclave_quote_body: body["isvEnclaveQuoteBody"].as_str().unwrap().to_owned(),
            revocation_reason: body["revocationReason"].as_str().map(b),
            pse_manifest_status: body["pseManifestStatus"].as_str().map(b),
            pse_manifest_hash: body["pseManifestHash"].as_str().map(b),
            platform_info_blob: body["platformInfoBlob"].as_str().map(b),
            nonce: body["nonce"].as_str().map(b),
            epid_pseudonym: body["epidPseudonym"].as_str().map(b),
        })
    }

    fn verify_response(
        root_ca_cert: &X509Cert,
        headers: &HeaderMap,
        body: &[u8],
    ) -> Result<(), AttestationError> {
        // Split certificates
        let re = Regex::new(
            "(-----BEGIN .*-----\\n)\
                            ((([A-Za-z0-9+/]{4})*\
                              ([A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)*\\n)+)\
                            (-----END .*-----)",
        )
        .unwrap();
        let (mut certificate, mut ca_certificate) = {
            let c = headers
                .get("x-iasreport-signing-certificate")
                .unwrap()
                .to_str()
                .unwrap();
            let c = percent_encoding::percent_decode_str(c)
                .decode_utf8()
                .unwrap();
            let c = re
                .find_iter(&c)
                .map(|m| m.as_str().to_owned())
                .collect::<Vec<String>>();
            let mut c_iter = c.into_iter();
            let mut certificate = c_iter.next().unwrap();
            certificate.push('\0');
            let certificate = X509Cert::new_from_pem(certificate.as_bytes()).unwrap();
            let mut ca_certificate = c_iter.next().unwrap();
            ca_certificate.push('\0');
            let ca_certificate = X509Cert::new_from_pem(ca_certificate.as_bytes()).unwrap();
            (certificate, ca_certificate)
        };

        // Check if the root certificate is the same as the SP-provided certificate
        if root_ca_cert != &ca_certificate {
            return Err(AttestationError::MismatchedIASRootCertificate);
        }

        // Check if the certificate is signed by root CA
        certificate
            .verify_this_certificate(&mut ca_certificate)
            .map_err(|_| AttestationError::InvalidIASCertificate)?;

        // Check if the signature is correct
        let signature = base64::decode(
            headers
                .get("x-iasreport-signature")
                .unwrap()
                .to_str()
                .unwrap(),
        )
        .unwrap();
        certificate
            .verify_signature(body, &signature[..])
            .map_err(|_| AttestationError::BadSignature)?;
        Ok(())
    }
}
