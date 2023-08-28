use mbedtls::ssl::config::{Endpoint, Preset, Transport};
use mbedtls::ssl::{Config, Context, HandshakeContext};
use mbedtls::Result;
use std::sync::Arc;
use super::random::Rng;

type Callback = Box<dyn FnMut(&mut HandshakeContext, &str) -> Result<()>>;

pub mod server {
    use super::*;
    pub fn callback(psk: &[u8]) -> Callback {
        let psk = psk.to_owned();
        Box::new(move |ctx: &mut HandshakeContext, _: &str| ctx.set_psk(psk.as_ref()))
    }

    pub fn config(rng: Rng, callback: &mut Callback) -> Config {
        let mut config = Config::new(Endpoint::Server, Transport::Stream, Preset::Default);
        config.set_rng(Arc::new(rng.inner));
        config.set_psk_callback(callback);
        config
    }

    pub fn context(config: Arc<Config>) -> Context {
        Context::new(config)
    }
}

pub mod client {
    use super::*;

    pub fn config(rng: Rng, psk: &[u8]) -> Result<Config> {
        let mut config = Config::new(Endpoint::Client, Transport::Stream, Preset::Default);
        config.set_rng(Arc::new(rng.inner));
        config.set_psk(psk, "Client_identity")?;
        Ok(config)
    }

    pub fn context(config: Arc<Config>) -> Context {
        Context::new(config)
    }
}
