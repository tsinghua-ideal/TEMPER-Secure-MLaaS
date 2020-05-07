# Secure-MLaaS
A Secure MLaaS Framework based on Intel SGX.
The framework contains two parts: model partition and model inferring.
## Brute Force 
Brute Force refers to a brute force searching algorithm to find best model partition.
A partition rule is applied, and the best situation of complexity is $$O(n)$$ while the worst is $$O(2^n)$$.
if 
$$Latency_1..(n+1)≤Latency_(1..n)+ Latency_(n+1)$$
, do not partition.
