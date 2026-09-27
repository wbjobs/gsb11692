# 流体/烟雾模拟 (Stable Fluids)

基于网格的简化 Navier-Stokes 求解器,双实现可切换对照:

- **WebGL2 (GPU)**:半拉格朗日平流 + Jacobi 压力投影,RGBA16F ping-pong 纹理
- **CPU (Web Worker)**:经典 Jos Stam stable fluids,Float32Array, transferable buffer 回传

## 运行

需要通过 HTTP 服务访问(ES Module + Worker 不支持 file://):

```bash
python3 -m http.server 8000
# 打开 http://localhost:8000
```

## 技术要点

| 关注点 | 处理方式 |
|---|---|
| 数值稳定性 | 半拉格朗日平流无条件稳定;dt 钳制 ≤ 1/30s;输入速度限幅;压力场每帧 ×0.8 衰减防误差累积 |
| 边界条件 | 速度无滑移(边界法向速度取反,GPU 专用 boundary pass / CPU setBnd);压力 Neumann(CLAMP_TO_EDGE / setBnd(0)) |
| 分辨率/性能取舍 | 64→1024 五档;GPU 端模拟与渲染解耦,统计面板实时显示 FPS/帧耗时/模拟耗时/网格规模/内存估算 |
| 大网格内存 | 全部 TypedArray/浮点纹理;1024² GPU ≈ 56MB 显存,CPU 模式上限 512² ≈ 12.6MB,面板实时估算 |
| 交互延迟 | pointermove 按帧合并为单次喷溅;CPU 模式 worker 未返回时不堆叠帧请求,喷溅入队随下一帧发送 |
| 降级策略 | 帧耗时 EMA > 30ms 持续 60 帧 → 自动降一档分辨率;已到最低档则减半压力迭代;WebGL 不可用/上下文丢失自动回退 CPU Worker |

## 性能观测

- `PerformanceObserver` 监听 `longtask`(>50ms 主线程阻塞计数与总时长)
- `performance.mark/measure` 包裹每帧模拟步,统计模拟耗时
- 勾选「自动降级」后可在高分辨率下观察降级事件日志,量化分辨率与帧率对照

## 文件结构

```
index.html          页面与控件
style.css           样式
js/main.js          编排: 交互、统计、降级、模式切换
js/fluid-gl.js      WebGL2 求解器
js/fluid-worker.js  CPU 求解器 (Web Worker)
```
