# pcap解析模块

## 思路

<img src="./assets/image-20260106101055858.png" alt="image-20260106101055858" style="zoom:67%;" />

## PCAPParser

解析原始pcap文件, 最终输出为`raw_packet`数据结构

可采用的方法:

- 基于 Scapy 的实现

