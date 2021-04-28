/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

 extern crate tvm_runtime;
 extern crate image;
 extern crate ndarray;
 extern crate rand;

 use rand::Rng;
 use std::{
     convert::TryFrom as _,
     io::{Read as _, Write as _},
     time::{SystemTime, UNIX_EPOCH},
     env,
 };
//  use image::{FilterType, GenericImageView};
 use ndarray::{Array, Array4};

 fn timestamp() -> i64 {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let ms = since_the_epoch.as_secs() as i64 * 1000i64 + (since_the_epoch.subsec_nanos() as f64 / 1_000_000.0) as i64;
    ms
}
 fn main() {
     env::set_var("TVM_NUM_THREADS", "6");

    let path = env!("PATH").split(":");
    let vec: Vec<&str> = path.collect();
    let a = vec[2].split("/");
    let vec: Vec<&str> = a.collect();
    let shape = (vec[0].to_string().parse::<usize>().unwrap(), vec[1].to_string().parse::<usize>().unwrap(), vec[2].to_string().parse::<usize>().unwrap(), vec[3].to_string().parse::<usize>().unwrap());
    // let sy_time = SystemTime::now();
    let syslib = tvm_runtime::SystemLibModule::default();
    let graph_json = include_str!(concat!(env!("OUT_DIR"), "/graph.json"));
    let params_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/params.bin"));
    let params = tvm_runtime::load_param_dict(params_bytes).unwrap();
     
    let graph = tvm_runtime::Graph::try_from(graph_json).unwrap();
    let mut exec = tvm_runtime::GraphExecutor::new(graph, &syslib).unwrap();
    exec.load_params(params);
    // println!("loading: {:?}", SystemTime::now().duration_since(sy_time).unwrap().as_micros());
    let mut rng =rand::thread_rng();
    let mut ran = vec![];
    for _i in 0..shape.0*shape.1*shape.2*shape.3{
        ran.push(rng.gen::<f32>()*256.);
    }
    let x = Array::from_shape_vec(shape, ran).unwrap();
    // println!("{:#?}", &x);
    
    let sy_time = SystemTime::now();
    exec.set_input("input.0", x.into());
    exec.run();
    let output = exec.get_output(0).unwrap();

    // let output = output.to_vec::<f32>();
    // find the maximum entry in the output and its index
    // let duration = SystemTime::now().duration_since(sy_time).unwrap().as_micros();
    // let mut argmax = -1;
    // let mut max_prob = 0.;
    // // println!("{:?}", output.len());
    // for i in 0..output.len() {
    //     if output[i] > max_prob {
    //         max_prob = output[i];
    //         argmax = i as i32;
    //     }
    // }
    // println!("It took {:?} us", duration);
     println!("{:?}", SystemTime::now().duration_since(sy_time).unwrap().as_micros());
    // println!("{:?}", duration);
    // let ts1 = timestamp();
    // println!("TimeStamp: {}", ts1);
    // println!("The index: {:?}", argmax);
    // println!("{:?}", sy_time.elapsed().unwrap().as_micros());
    // println!("{:#?}", output.data().as_slice());

 }
 