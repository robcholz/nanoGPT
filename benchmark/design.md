# Design

We collect the following data:

- timestamp
- step
- phase
  - forward 
  - backward
  - optimizer 
  - eval
  - checkpoint 
  - dataloader 
  - idle
- interval_s
- cpu_util_percent
- gpu_util_percent
- gpu_mem_mb
- gpu_power_w
- host_mem_mb
- disk_read_mb_s
- disk_write_mb_s

Sampling frequency is 100ms.
