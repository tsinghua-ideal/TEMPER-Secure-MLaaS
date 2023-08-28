use serde_json::{Result, Value};
use serde::{Serialize, Deserialize};
#[derive(Serialize, Deserialize)]
// This is the source file of scheduler.
pub struct Slave {
    pub busy_flag: bool,
    pub slave_ip: String, 
} 

impl Clone for Slave {
    fn clone(&self) -> Self {
        Self { busy_flag: self.busy_flag, slave_ip: self.slave_ip.clone() }
    }
}
pub struct User {
    pub sub_model: Vec<String>,
    pub model: String, 
}
impl Clone for User {
    fn clone(&self) -> Self {
        let mut sb: Vec<String> = vec![];
        for iter in self.sub_model.clone(){
            sb.push(iter);
        }
        Self { sub_model: sb, model: self.model.clone() }
    }
}
pub struct Scheduler {
    pub map_table: Value,
    // addresses of users
    pub user_queue: Vec<User>,
    // addresses of slaves
    pub slave_queue: Vec<Slave>,
}
impl Scheduler {
    fn clone(&self) -> Self {
        Self { map_table: self.map_table.clone(), user_queue: self.user_queue.clone(), slave_queue: self.slave_queue.clone()}
    }
    // fn model2ip(self, model: &a' str) -> &a' str {
    //     let ip = match model {
    //         '0' => "127.0.0.1:4242",
    //         '1' => "127.0.0.1:4243",
    //         '2' => "127.0.0.1:4244",
    //         '3' => "127.0.0.1:4245",
    //         '4' => "127.0.0.1:4246",
    //         _ => "127.0.0.1:4242",
    //     }
    //     ip
    // }
    // Initialize the mapping table.
    // Maybe there is an easier way for configuraration loading.
    pub fn init(self, config: Value) -> Scheduler {
        // let data: &str = r#"
        // {
        //     "server_address": "127.0.0.1:22000",
        //     "client_address": "127.0.0.1:4240",
        //     "attestation_address": "127.0.0.1:4240",
        //     "model": ["resnet18", "mobilenetv1"],
        //     "slave_address": {
        //         "resnet18": {
        //             "slave": {
        //                 "0": "127.0.0.1:32100",
        //                 "1": "127.0.0.1:4241",
        //                 "2": "127.0.0.1:4242",
        //                 "3": "127.0.0.1:4243",
        //                 "4": "127.0.0.1:4244"
        //             }
        //         },
        //         "mobilenetv1": {
        //             "slave": {
        //                 "0": "127.0.0.1:4250",
        //                 "1": "127.0.0.1:4251",
        //                 "2": "127.0.0.1:4252",
        //                 "3": "127.0.0.1:4253",
        //                 "4": "127.0.0.1:4254",
        //                 "5": "127.0.0.1:4255"
        //             }
        //         }
        //     }
        // }"#;
        // let config:  Value = serde_json::from_str(data).unwrap();
        let map_table = config.clone();
        let user_queue: Vec<User> = vec![];
        let mut slave_queue: Vec<Slave> = vec![];
        for i in 0..map_table["slave_address"]["resnet18"]["slave"].as_object().unwrap().len(){
            slave_queue.push(Slave{busy_flag: false, slave_ip: config["slave_address"]["resnet18"]["slave"][i.to_string()].as_str().unwrap().to_string().clone()})
        }
        for i in 0..map_table["slave_address"]["mobilenetv1"]["slave"].as_object().unwrap().len(){
            slave_queue.push(Slave{busy_flag: false, slave_ip: config["slave_address"]["mobilenetv1"]["slave"][i.to_string()].as_str().unwrap().to_string().clone()})
        }
        // let slave_queue: Vec<Slave> = vec![Slave{busy_flag: false, slave_ip: "127.0.0.1:4242"}, Slave{busy_flag: false, slave_ip: "127.0.0.1:4243"}, 
        // Slave{busy_flag: false, slave_ip: "127.0.0.1:4244"}, Slave{busy_flag: false, slave_ip: "127.0.0.1:4245"}, Slave{busy_flag: false, slave_ip: "127.0.0.1:4246"},
        // Slave{busy_flag: false, slave_ip: "127.0.0.1:4247"}, Slave{busy_flag: false, slave_ip: "127.0.0.1:4248"}];
        Scheduler {map_table, user_queue, slave_queue}        
    }
    pub fn is_slave_busy(self, slave_ip: String) -> Result<(Slave, bool)>{
        let slave = Slave{busy_flag: false, slave_ip: "".to_string()};
        for slv in self.slave_queue{
            if slv.slave_ip == slave_ip {
                let result = match slv.busy_flag {
                    true => Ok((slv, true)),
                    false => Ok((slv, false)),
                };
                // println!("{:?}", false);
                return result;
            }
        }
        Ok((slave, false))
    }
    pub fn add_user(&mut self, model: String) -> bool{
        // let mut sub_model: Vec<String> = vec![];
        // let user_ip = user_ip;
        // let mut user = User{sub_model, user_ip};
        let users = self.user_queue.clone();
        for usr in users{
            if usr.sub_model.len() > 0 && model == usr.model{
                return false;
            }
        }
        // println!("{:?}", model);
        let m = model.clone();
        let m: Vec<&str> = m.split(',').collect();
        let m = m[0].to_string();
        for md in self.map_table.clone()["model"].as_array().unwrap(){
            let md = md.as_str().unwrap();
            if md == m.clone().as_str(){
                // check_user_not_existed(user_ip);
                let mut sub_model:Vec<String> = vec![];
                let len = self.map_table.clone()["slave_address"][m.clone()]["slave"].as_object().unwrap().len();
                for i in 0..len{
                    sub_model.push(i.to_string());
                    
                }
                // sub_model.push("-1".to_string());   //as a flag of eof
                let user = User{sub_model, model: model.clone()};
                self.user_queue.push(user);
                // let p: Vec<&str> = md.1.split(',').collect();
                // println!("{:?}", p);
            }
        }
        return true;
    }
    pub fn change_slave_flag(&mut self, slave_ip: String){
        for i in 0..self.slave_queue.len(){
            if self.slave_queue[i].slave_ip == slave_ip
            {
                self.slave_queue[i].busy_flag = match self.slave_queue[i].busy_flag{
                    false => true,
                    true => false,
                };
            }
        }
    }
    pub fn apply4slave(&mut self, model: String, last_slave_ip: String) -> String{
        let mut ip = "".to_string();
        let mut prefix = "";
        for i in 0..self.user_queue.len(){
            let usr = self.user_queue[i].clone();
            if model == usr.model{
                let sc = self.clone();

                let m = model.clone();
                let m: Vec<&str> = m.split(',').collect();
                let m = m[0].to_string();
                println!("{:?}", usr.sub_model[0].clone());
                ip = self.map_table["slave_address"][m]["slave"][usr.sub_model[0].clone()].as_str().unwrap().to_string();
                let (slv, result) = sc.is_slave_busy(ip.clone()).unwrap();
                println!("{:?}", last_slave_ip);
                if result {
                    return ip.clone().to_string();
                }
                ip = slv.slave_ip.clone();
                self.change_slave_flag(slv.slave_ip.clone());
                self.change_slave_flag(last_slave_ip.clone());
                self.user_queue[i].sub_model.remove(0);
                println!("len {:?}", self.user_queue[i].sub_model.len());
                if self.user_queue[i].sub_model.len() == 0{
                    // println!("inferring end");
                    self.user_queue.remove(i);
                    prefix = "finished";
                }
                // println!("hit: {:?}", self.slave_queue[0].busy_flag);
            }
        }
        format!("{},{}",prefix, ip)
    }
    #[allow(dead_code)]
    pub fn user_success(mut self, slave_ip: String){
        self.change_slave_flag(slave_ip);
    }
}