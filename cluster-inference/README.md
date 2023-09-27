# cluster-inference
For safety concern, we have to build a reliable communication architecture to ensure that the service is correctly initialized in the cloud and not modified and the secure point-to-point channel is built between user and Secure MLaaS. 


## Port distribution
| Name        | ip    |  port(communication\|attestation)  |
| --------   | -----:   | :----: |
| Slave        |    IP   |   22100---, 32100---    |
| scheduler        |   IP    |   22000, 32000   |
| Overlay        |   IP    |   21000---, 31000---    |


## Command
Run `run.sh` to start the service, or run the following commands step by step.
For example, 
```shell
python worker_generator.py /home/test/model/resnet50 /home/test/Secure-MLaaS/cluster-inference/ae
cd /home/test/Secure-MLaaS/cluster-inference/
./build.sh ae/
./run.sh ae/
```