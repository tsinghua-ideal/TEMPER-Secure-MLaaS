#!/bin/bash
#sev=`ftxsgx-runner part1/target/x86_64-fortanix-unknown-sgx/debug/sgx-demo.sgxs`
#cli=`ftxsgx-runner part0/target/x86_64-fortanix-unknown-sgx/debug/sgx-demo.sgxs`
file=full_transition.txt
for i in {0..50};
do
    echo $i
    echo `ftxsgx-runner part1/target/x86_64-fortanix-unknown-sgx/debug/sgx-demo.sgxs` >> ${file} &
    sleep 0.5
    echo `ftxsgx-runner part0/target/x86_64-fortanix-unknown-sgx/debug/sgx-demo.sgxs` >> ${file}
done;