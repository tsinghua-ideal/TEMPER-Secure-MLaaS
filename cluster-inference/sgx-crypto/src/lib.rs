pub mod aes_gcm;
pub mod certificate;
pub mod cmac;
pub mod digest;
pub mod error;
pub mod key_exchange;
pub mod random;
pub mod signature;
pub mod tls_psk;
pub mod keys;
pub type Result<T> = std::result::Result<T, error::CryptoError>;

extern crate cfg_if;