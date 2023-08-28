use mbedtls::cipher::raw::{Cipher, CipherId, CipherMode};

pub const MAC_LEN: usize = 16;
pub type MacTag = [u8; MAC_LEN];

/// 128-bit AES-CMAC
pub struct Cmac {
    inner: Cipher,
    key: [u8; MAC_LEN],
}

impl Cmac {
    pub fn new(key: &[u8]) -> super::Result<Self> {
        let mut _key = [0u8; MAC_LEN];
        _key.as_mut().clone_from_slice(key);
        Ok(Self {
            inner: Cipher::setup(CipherId::Aes, CipherMode::ECB, (MAC_LEN * 8) as u32)?,
            key: _key,
        })
    }

    pub fn sign(&mut self, data: &[u8]) -> super::Result<MacTag> {
        let mut tag = [0u8; MAC_LEN];
        self.inner.cmac(self.key.as_ref(), data, tag.as_mut())?;
        Ok(tag)
    }

    pub fn sign_len(&mut self, data: &mut [u8], len: usize) -> super::Result<Vec<u8>> {
        let mut tag = [0u8; MAC_LEN];
        let cmac_len = MAC_LEN as usize;
        let mut result = Vec::new();
        let rang = len / cmac_len;
        for _i in 0..rang {
            self.inner.cmac(self.key.as_ref(), data, tag.as_mut())?;
            result.extend_from_slice(&tag);
            self.key[..].clone_from_slice(&tag);
        }
        let fi = len % cmac_len;
        if fi != 0 {
            self.inner.cmac(self.key.as_ref(), data, tag.as_mut())?;
            result.extend_from_slice(&tag[0..fi]);
        }
        Ok(result)
    }

    pub fn verify(&mut self, data: &[u8], tag: &MacTag) -> super::Result<()> {
        let ref_tag = self.sign(data)?;
        match &ref_tag == tag {
            true => Ok(()),
            false => Err(super::error::CryptoError::CmacVerificationError),
        }
    }
}
