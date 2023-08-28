// pub use mbedtls::rng::EntropyCallback;

#[cfg(not(target_env = "sgx"))]
use std::sync::Arc;

cfg_if::cfg_if! {
    if #[cfg(any(feature = "rdrand", target_env = "sgx"))] {
        pub fn entropy_new() -> mbedtls::rng::Rdseed {
            mbedtls::rng::Rdseed
        }
    } else if #[cfg(feature = "std")] {
        pub fn entropy_new() -> crate::mbedtls::rng::OsEntropy {
            crate::mbedtls::rng::OsEntropy::new()
        }
    } else {
        pub fn entropy_new() -> ! {
            panic!("Unable to run test without entropy source")
        }
    }
}


#[cfg(target_env = "sgx")]
pub struct Rng {
    pub inner: mbedtls::rng::Rdrand,
}

#[cfg(not(target_env = "sgx"))]
pub struct Rng {
    pub inner: mbedtls::rng::CtrDrbg,
}

#[cfg(target_env = "sgx")]
impl Rng {
    pub fn new() -> Self {
        Self {
            inner: mbedtls::rng::Rdrand,
        }
    }
}

#[cfg(not(target_env = "sgx"))]
impl Rng {
    pub fn new() -> Self {
        let entropy = Arc::new(mbedtls::rng::OsEntropy::new());
        Self {
            inner: mbedtls::rng::CtrDrbg::new(entropy, None).unwrap(),
        }
    }
}
