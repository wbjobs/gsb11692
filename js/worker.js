/**
 * worker.js — 模拟线程。
 *
 * 职责:
 *  - 固定步长推进 FluidSolver (accumulator, 最多 2 个子步, 防止"死亡螺旋")
 *  - 接收主线程的 splat / 参数 / 分辨率消息
 *  - 每帧把密度场转成 Uint8 (0..255) 通过 Transferable ArrayBuffer 发回主线程
 *  - 双缓冲: 主线程渲染完把 buffer 还回来 (recycle), 全程零分配
 *  - 上报每步模拟耗时 (simMs), 供主线程做性能对照
 */
import { FluidSolver } from './solver.js';

const FIXED_DT = 1 / 60;
const MAX_SUBSTEPS = 2;

let solver = null;
let params = { visc: 0.0001, diff: 0.00001, vorticity: 2.0, fade: 0.15 };
let running = false;
let timer = null;
let lastTime = 0;
let accumulator = 0;
let stepCount = 0;
let simMsEMA = 0;

// 双缓冲: 两块可转移的 Uint8 帧缓冲轮流使用
let frameBuffers = [];
let pendingSplats = [];

function allocFrameBuffers(N) {
  frameBuffers = [new Uint8Array(N * N), new Uint8Array(N * N)];
}

function init(N) {
  solver = new FluidSolver(N);
  allocFrameBuffers(N);
  pendingSplats = [];
  accumulator = 0;
  stepCount = 0;
  postMessage({
    type: 'ready',
    N,
    memoryBytes: solver.memoryBytes() + N * N * 2,
  });
}

function tick(now) {
  if (!running || !solver) return;
  // 帧间隔钳制在 [0, 50ms]: 切后台回来不会产生巨大 dt
  let frameDt = Math.min((now - lastTime) / 1000, 0.05);
  lastTime = now;
  accumulator += frameDt;

  let substeps = 0;
  const t0 = performance.now();
  while (accumulator >= FIXED_DT && substeps < MAX_SUBSTEPS) {
    // 先应用本帧所有交互 splat (批量, 降低消息延迟)
    for (const s of pendingSplats) {
      solver.splat(s.x, s.y, s.radius, s.fx, s.fy, s.dye);
    }
    pendingSplats.length = 0;
    solver.step(FIXED_DT, params);
    accumulator -= FIXED_DT;
    substeps++;
    stepCount++;
  }
  if (substeps === MAX_SUBSTEPS) accumulator = 0; // 跟不上就丢时间, 不追帧
  const simMs = performance.now() - t0;
  simMsEMA = simMsEMA * 0.9 + simMs * 0.1;

  // 有可用帧缓冲才渲染输出, 否则跳帧 (背压保护)
  const buf = frameBuffers.pop();
  if (buf) {
    const N = solver.N;
    const dens = solver.dens;
    const stride = N + 2;
    for (let j = 0; j < N; j++) {
      const srcRow = (j + 1) * stride + 1;
      const dstRow = j * N;
      for (let i = 0; i < N; i++) {
        const d = dens[srcRow + i];
        buf[dstRow + i] = d >= 1 ? 255 : (d * 255) | 0;
      }
    }
    postMessage({
      type: 'frame',
      buffer: buf.buffer,
      N,
      simMs: simMsEMA,
      stepCount,
    }, [buf.buffer]);
  }
}

function start() {
  if (timer) return;
  running = true;
  lastTime = performance.now();
  timer = setInterval(() => tick(performance.now()), 1000 / 60);
}

function stop() {
  running = false;
  if (timer) { clearInterval(timer); timer = null; }
}

onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init':
      init(msg.N);
      start();
      break;
    case 'resize':
      stop();
      init(msg.N);
      start();
      break;
    case 'params':
      Object.assign(params, msg.params);
      break;
    case 'splat':
      // 主线程已按帧批量合并, 这里只入队
      pendingSplats.push(msg.splat);
      break;
    case 'recycle':
      // 主线程归还渲染完的 buffer
      frameBuffers.push(new Uint8Array(msg.buffer));
      break;
    case 'pause':
      stop();
      break;
    case 'resume':
      start();
      break;
    case 'clear':
      if (solver) solver.reset();
      break;
  }
};
