# 数据流层次设计(EFSM构建)

## 总体流程

<img src="./assets/image-20260301182513173.png" alt="image-20260301182513173" style="zoom:67%;" />

## 设计思路

### 得到会话内容指标信息

首先得到`SessionContext`, 即记录会话特征的容器, 其数据结构为:

```python
@dataclass
class SessionContext:

    # 方向判定变量
    is_client: bool = False
    server_port: int = 0
    
    # 统计变量
    packet_ratio: float = 0.0  # C2S / S2C 报文比例
    byte_ratio: float = 0.0    # C2S / S2C 字节比例
    pair_ratio: float = 0.0    # 请求-响应配对比例
    
    # 协议特征
    avg_message_len: float = 0.0
    message_len_std: float = 0.0
    total_messages: int = 0
    
    # 时序特征
    avg_interval: float = 0.0
    burstiness: float = 0.0
    
    # 协议指纹
    is_request_response: bool = False
    is_streaming: bool = False
    suspected_protocol: str = ""
```

#### 统计变量

- `packet_ratio`:

  客户端**发包数** / 服务器发包数, 如http中总是请求短, 响应的长, 则此时<1

- `byte_ratio`:

  客户端**字节数** / 服务器字节数

- `pair_ratio`:

  请求-响应成功配对比例, 如高的则为典型的请求-响应协议, 低则为流式协议

### 特征提取

默认采用Kmeans聚类获取特征, 可以传入其他的聚类模型, 将
