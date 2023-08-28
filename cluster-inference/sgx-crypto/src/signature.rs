use super::digest::{sha256, SHA256_TYPE};
use super::random::Rng;
use mbedtls::ecp::{EcGroup, EcPoint};
use mbedtls::pk::{EcGroupId, Pk, ECDSA_MAX_LEN};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub type ECDSAPublicKey = Vec<u8>;
pub type Signature = Vec<u8>;
const ECGROUP_ID: EcGroupId = EcGroupId::SecP256K1;
pub struct VerificationKey {
    inner: Pk,
}

impl VerificationKey {
    /// Takes both DER and PEM forms of PKCS#1 or PKCS#8 encoded keys.
    /// When calling on PEM-encoded data, key must be NULL-terminated
    pub fn new(public_key: &[u8]) -> super::Result<Self> {
        let inner = Pk::from_public_key(public_key)?;
        Ok(Self { inner })
    }
    pub fn new_from_binary(public_key: &[u8]) -> super::Result<Self> {
        let secp256k1 = EcGroup::new(ECGROUP_ID).unwrap();
        let rec_pt = EcPoint::from_binary(&secp256k1, &public_key).unwrap();
        let inner = Pk::public_from_ec_components(secp256k1, rec_pt).unwrap();
        Ok(Self { inner })
    }

    /// Takes both DER and PEM forms of PKCS#1 or PKCS#8 encoded keys.
    pub fn new_from_file(public_key_path: &Path) -> super::Result<Self> {
        let mut file = File::open(public_key_path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        if public_key_path.extension().unwrap() == "pem" {
            buf.push(0);
        }
        Self::new(&buf[..])
    }

    pub fn verify(&mut self, message: &[u8], signature: &[u8]) -> super::Result<()> {
        let hash = sha256(message)?;
        self.inner.verify(SHA256_TYPE, &hash[..], signature)?;
        Ok(())
    }
}

pub struct SigningKey {
    inner: Pk,
}
impl SigningKey {
    /// Takes both DER and PEM forms of PKCS#1 or PKCS#8 encoded keys.
    /// When calling on PEM-encoded data, key must be NULL-terminated
    pub fn new(private_key: &[u8], password: Option<&[u8]>) -> super::Result<Self> {
        let inner = Pk::from_private_key(private_key, password)?;
        Ok(Self { inner })
    }
    pub fn generate_keypair(rng: &mut Rng) -> super::Result<Self> {
        Ok(Self {
            inner: Pk::generate_ec(&mut rng.inner, ECGROUP_ID)?,
        })
    }

    pub fn get_public_key(&self) -> super::Result<ECDSAPublicKey> {
        let ecgroup = self.inner.ec_group()?;
        Ok(self.inner.ec_public()?.to_binary(&ecgroup, true)?)
    }
    /// Takes both DER and PEM forms of PKCS#1 or PKCS#8 encoded keys.
    pub fn new_from_file(private_key_path: &Path, password: Option<&[u8]>) -> super::Result<Self> {
        let mut file = File::open(private_key_path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        if private_key_path.extension().unwrap() == "pem" {
            buf.push(0);
        }
        Self::new(&buf[..], password)
    }

    pub fn sign(&mut self, message: &[u8], rng: &mut Rng) -> super::Result<Signature> {
        let hash = sha256(message)?;
        let sig_len = self.inner.rsa_public_modulus()?.byte_length()?;
        let mut signature = vec![0u8; sig_len];
        self.inner
            .sign(SHA256_TYPE, &hash[..], &mut signature[..], &mut rng.inner)?;
        Ok(signature)
    }
    pub fn ecdsa_sign(&mut self, message: &[u8], rng: &mut Rng) -> super::Result<Signature> {
        let hash = sha256(message)?;
        let sig_len = ECDSA_MAX_LEN;
        let mut signature = vec![0u8; sig_len];
        let len = self.inner.sign_deterministic(
            SHA256_TYPE,
            &hash[..],
            &mut signature[..],
            &mut rng.inner,
        )?;
        //Ok(signature)
        Ok(signature[..len].to_vec())
    }
}
