use std::io::{Write, Read, BufReader};
use byteorder::{NetworkEndian, ReadBytesExt};
use std::time;
use std::thread;
use crate::thread_client::TaskEnclave;
use mbedtls::ssl::{Config, Context};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock, Arc};
pub static IMAGE_SIZE:usize=3*224*224;
pub static ALLOW_BATCH_SIZE:usize=4;
lazy_static! {
    pub static ref USER_DATAS: Mutex<HashMap<String, Vec<u8>> > = Mutex::new(HashMap::new());
    pub static ref RESULTS: RwLock<HashMap<String, Vec<u8>> > = RwLock::new(HashMap::new());
    pub static ref TENCLAVES: Arc<RwLock<Vec<TaskEnclave>>> = Arc::new(RwLock::new(Vec::new()));
}
pub fn task_schedule(rc_config:Arc<Config>){

    let mut own_ud_map = HashMap::with_capacity(8);

    loop {        
        let ens = TENCLAVES.read().unwrap();
        if ens.is_empty() {
            let ten_millis = time::Duration::from_millis(1000);
            thread::sleep(ten_millis);
            continue;
        }
        // println!("enclave number is {:?}", ens.len());      
       {     
           let mut user_datas = USER_DATAS.lock().unwrap();
            // println!("user number is {:?}", user_datas.len());
            if ens.is_empty()  || user_datas.is_empty() {           
                let ten_millis = time::Duration::from_millis(1000);
                thread::sleep(ten_millis);
                continue;
            }        
            let mut keys = Vec::new();   
            if own_ud_map.len()<=16{
                for key in (*user_datas).keys() {
                    keys.push(key.clone());
                    if keys.len()>=16-own_ud_map.len(){
                        break;
                    }
                }
                for k in keys{
                    let (key, val )=(*user_datas).remove_entry(&k).unwrap();
                    if (val.len() % IMAGE_SIZE) ==0{
                        own_ud_map.insert(key, val);
                    }else {
                        let mut resultsmap=RESULTS.write().unwrap();        
                        resultsmap.insert(key.clone(), vec![0u8;4]);
                    }

                }
            }
        }
        let mut keys = Vec::new();
        let mut all_len =0usize;
        for (k,v ) in own_ud_map.iter(){
            keys.push(k.clone());
            all_len +=v.len()/IMAGE_SIZE;
        }
        //must be n times of ALLOW_BATCH_SIZE, such 4,8,12,16
        let all_len = (all_len +ALLOW_BATCH_SIZE-1)/ALLOW_BATCH_SIZE;
        if own_ud_map.is_empty() || all_len < 1{
            let ten_millis = time::Duration::from_millis(1000);
            thread::sleep(ten_millis);
            continue;
        }
        let mut ctx = Context::new(Arc::clone(&rc_config));
        let mut io=(*ens)[0].get_stream().try_clone().unwrap();
        let mut v_msg =vec![8u8; 8];
        v_msg.push(all_len as u8);
        io.write(&v_msg).unwrap();
        let mut msg = vec![0u8; 23];
        while msg[0..8] != [8u8; 8] {
            io.read(&mut msg).unwrap();
            println!(
                "msg {:#?}",
                msg.clone()
                    .into_iter()
                    .map(|x| format!(" {:x}", x))
                    .collect::<String>()
            );
        }
        println!("New a TLS connection to enclave, enclave addr: {:#?}", (*ens)[0].get_stream().peer_addr().unwrap().to_string());   
        ctx.establish( io, None).unwrap();
        let mut server_session = BufReader::new(ctx);
        let mut j =0usize; //who's data
        let mut k=0usize;  //user data index
        let mut v_n = Vec::new();
        for _i in 0..all_len{
            v_n.clear();
            let mut n =0usize; //which batch
            while n < ALLOW_BATCH_SIZE{
                let name =&keys[j];
                println!(" {:?}", name);
                let data = own_ud_map.get(name).unwrap();
                if data.len()/IMAGE_SIZE <= k+ALLOW_BATCH_SIZE-n{
                    v_n.append(&mut vec![name.clone(); data.len()/IMAGE_SIZE-k]);
                    let data =own_ud_map.remove(name).unwrap();
                    server_session.get_mut().write_all(&data[k*IMAGE_SIZE..data.len()]).unwrap();
                    // own_ud_map.remove(name);
                    j=j+1;                    
                    n=n+data.len()/IMAGE_SIZE-k;
                    k=0;
                    if j >= keys.len(){
                        server_session.get_mut().write_all(&vec![0u8; (ALLOW_BATCH_SIZE-n)*IMAGE_SIZE]).unwrap();
                        n=4;
                    }                                                        
                }else {
                    
                    v_n.append(&mut vec![name.clone(); ALLOW_BATCH_SIZE-n]);
                    server_session.get_mut().write_all(&data[k*IMAGE_SIZE..(k+ALLOW_BATCH_SIZE-n)*IMAGE_SIZE]).unwrap();
                    k=k+ALLOW_BATCH_SIZE-n;
                    n=ALLOW_BATCH_SIZE;                    
                }
            }

            let result_len = server_session.read_u32::<NetworkEndian>().unwrap() as usize;
            let mut classfic =vec![0u8;result_len];

            server_session.read_exact(&mut classfic).unwrap();           

            println!("Last 16bytes of output: {:?}", &classfic[classfic.len()-16..classfic.len()]);       
            println!("Insert result to output map, len is {:?}!", classfic.len());
            {
                v_n.push("endl".to_string());
                let mut resultsmap=RESULTS.write().unwrap();
                let result_size = result_len/ALLOW_BATCH_SIZE;
               
                let mut s_index=0usize;
                let mut h_name =&v_n[0];
                for (index, name) in v_n.iter().enumerate(){
                    if !name.eq(h_name) {
                        // println!("s_index {} {:?} end_index {}  {:?}", s_index,h_name, index, name);
                        let mut data =   classfic[s_index*result_size..index*result_size].to_vec();
                        if resultsmap.contains_key(h_name){
                            let mut res = resultsmap.remove(name).unwrap();
                            res.append(&mut data);
                            resultsmap.insert(name.clone(), res);
                        } else {
                            resultsmap.insert(h_name.clone(), data);
                        }
                        s_index = index;
                        h_name=name;
                    }
                }
            }
        }
        server_session.get_mut().close();
        println!("Finish a task!");
    }    
    
}