## Machine used:
1 x A100 (40 GB SXM4)
30 vCPUs, 200GiB RAM, 512 GiB SSD

## Pipeline run with models cached + float32
Pipeline Runtime: 65.6 s
Average GPU usage: 70%
Peak GPU usage: 97%
Average GPU memory usage: 10.14/40Gi = 25.4%
Peak GPU memory usage: 12.33/40Gi = 30.8%

## Pipeline run with models cached + float16
Pipeline Runtime: 52.9 s
Average GPU usage: 52%
Peak GPU usage: 88%
Average GPU memory usage: 6.10/40Gi = 15.3%
Peak GPU memory usage: 7.49/40Gi = 18.7%

## Pipeline run with models cached + float16 + memory efficient settings:
Pipeline Runtime: 186.6 s
Average GPU usage: 47%
Peak GPU usage: 62%
Average GPU memory usage: 2.55/40Gi = 6.4%
Peak GPU memory usage: 4.63/40Gi = 11.5%

## Pipeline run with models cached + float16 + memory efficient settings + garbage collection off:
Pipeline Runtime: 181.9 s
Average GPU usage: 45%
Peak GPU usage: 61%
Average GPU memory usage: 2.55/40Gi = 6.38%
Peak GPU memory usage: 4.63/40Gi = 11.5%

## Average model load time from cached dir
2468 ms
