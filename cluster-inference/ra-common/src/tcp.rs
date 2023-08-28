use std::io::{Error, ErrorKind, Result};
use std::net::{TcpListener, TcpStream};
use std::thread::sleep;
use std::time::{Duration, Instant};

const CONNECT_SLEEP_TIME_MILLIS: u64 = 10;

pub fn tcp_connect(addr: &str, timeout: Duration) -> Result<TcpStream> {
    let start = Instant::now();
    loop {
        match TcpStream::connect(addr) {
            Ok(s) => {
                return Ok(s);
            }
            Err(e) => {
                if start.elapsed() == timeout {
                    return Err(Error::new(ErrorKind::TimedOut, e));
                }
            }
        }
        sleep(Duration::from_millis(CONNECT_SLEEP_TIME_MILLIS));
    }
}

pub fn tcp_accept(addr: &str) -> Result<TcpStream> {
    let listener = TcpListener::bind(addr)?;
    Ok(listener.accept()?.0)
}
