use mbedtls::cipher::raw::{Cipher, CipherId, CipherMode, Operation::Encrypt};

pub const MAC_LEN: usize = 16;
pub type MacTag = [u8; MAC_LEN];

/// 256-bit AES-GCM
pub struct AESGCM {
    inner: Cipher,
}

impl AESGCM {
    // default key len == 32 bytes
    pub fn new() -> super::Result<Self> {
        // _key.as_mut().clone_from_slice(key);
        Ok(Self {
            inner: Cipher::setup(CipherId::Aes, CipherMode::GCM, (32 * 8) as u32)?,
        })
    }
    pub fn new_with_key(key: &[u8], key_len: u32) -> super::Result<Self> {
        // _key.as_mut().clone_from_slice(key);
        if key_len != 16 && key_len != 24 && key_len != 32 {
            panic!("error key len for aes gcm mode");
        }
        let mut inner = Cipher::setup(CipherId::Aes, CipherMode::GCM, (key_len * 8) as u32)
            .expect("Cipher::setup error");
        // inner.set_key_and_maybe_iv(key, None)?;
        inner.set_key(Encrypt, key).expect("Cipher::set_key error");
        let iv = [0u8; 16];
        inner.set_iv(&iv).expect("Cipher::set_iv error");
        inner.reset()?;
        Ok(Self { inner })
    }
    pub fn set_key_iv(&mut self, key: &[u8], iv: &[u8]) -> super::Result<()> {
        // self.inner.set_key_and_maybe_iv(key, Some(iv))?;
        self.inner.set_key(Encrypt, key)?;
        self.inner.set_iv(iv)?;
        self.inner.reset()?;
        Ok(())
    }
    pub fn encrypt(
        &mut self,
        plain_text: &[u8],
        cipher_and_tag: &mut [u8],
        tag_len: usize,
    ) -> super::Result<usize> {
        let ad = [0u8, 0, 0, 0];
        // let cipher_and_tag = cipher_text. + &tag;
        let len = self
            .inner
            .encrypt_auth(&ad, plain_text, cipher_and_tag, tag_len)
            .expect("Cipher::encrypt_auth");
        Ok(len)
    }

    pub fn decrypt(
        &mut self,
        cipher_text_and_tag: &[u8],
        plain_text: &mut [u8],
        tag_len: usize,
    ) -> super::Result<usize> {
        let ad = [0u8, 0, 0, 0];
        Ok(self.inner.decrypt_auth(&ad, cipher_text_and_tag, plain_text, tag_len)?)
    }
}
