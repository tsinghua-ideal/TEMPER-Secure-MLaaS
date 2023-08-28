#!/bin/bash

cat /proc/`ps -ef | grep sgx-demo.sgxs | grep -v grep |awk '{print $2}'`/status |grep -i VM

rm -rf log.txt
`ftxsgx-runner target/x86_64-fortanix-unknown-sgx/debug/sgx-demo.sgxs` &

while(expr `ps -ef | grep sgx-demo.sgxs|grep -v grep |awk '{print $2}'`)
do {
    # cat /proc/`ps -ef | grep sgx-demo.sgxs|grep -v grep |awk '{print $2}'`/status |grep -i VM >>log.txt
    cat /proc/`ps -ef | grep sgx-demo.sgxs|grep -v grep |awk '{print $2}'`/status >>log.txt
    echo "----------------------------" >>log.txt
}
done
echo "finished!"

