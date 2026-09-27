/**
 * main.js — 主线程: UI、交互、性能监控、自动降级。
 *
 * 性能策略:
 *  - 模拟在 Web Worker, 渲染在 WebGL, 主线程只做合成与 UI
 *  - rAF 测 FPS (EMA), PerformanceObserver 监听 longtask 统计卡顿
 *  - 自动降级: 连续低 FPS -> 降低网格分辨率; 恢复后不回弹 (避免抖动)
 *  - 交互: pointermove 采样合并, 每帧最多发一条批量 splat 消息 (降低延迟与消息洪泛)
 */
import { Renderer } from './renderer.js';

const RESOLUTIONS = [64, 96, 128, 192, 256];
const MIN_AUTO_RES = 64;

const canvas = document.getElementById('view');
const ui = {
  resolution: document.getElementById('resolution'),
  viscosity: document.getElementById('viscosity'),
  diffusion: document.getElementById('diffusion'),
  force: document.getElementById('force'),
  dye: document.getElementById('dye'),
  vorticity: document.getElementById('vorticity'),
  autoDegrade: document.getElementById('autoDegrade'),
  pause: document.getElementById('pause'),
  clear: document.getElementById('clear'),
  fps: document.getElementById('fps'),
  simMs: document.getElementById('simMs'),
  grid: document.getElementById('grid'),
  mem: document.getElementById('mem'),
  renderMode: document.getElementById('renderMode'),
  longtasks: document.getElementById('longtasks'),
  steps: document.getElementById('steps'),
  degraded: document.getElementById('degraded'),
};

const renderer = new Renderer(canvas);
ui.renderMode.textContent = renderer.mode;

let worker = null;
let N = 128;
let paused = false;
let latestFrame = null;   // {buffer, N, simMs, stepCount}
let fpsEMA = 0;
let lastRAF = performance.now();
let longtaskCount = 0;
let degradedNotice = '';

// ---- PerformanceObserver: 长任务监控 (>50ms 的主线程阻塞) ----
if ('PerformanceObserver' in window) {
  try {
    new PerformanceObserver((list) => {
      longtaskCount += list.getEntries().length;
      ui.longtasks.textContent = String(longtaskCount);
    }).observe({ entryTypes: ['longtask'] });
  } catch { /* longtask 不可用时静默 */ }
}

// ---- Worker ----
function spawnWorker(res) {
  if (worker) worker.terminate();
  worker = new Worker('./js/worker.js', { type: 'module' });
  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'ready') {
      ui.grid.textContent = `${msg.N} × ${msg.N} (${(msg.N + 2)}² 含边界)`;
      ui.mem.textContent = (msg.memoryBytes / 1024 / 1024).toFixed(2) + ' MB';
    } else if (msg.type === 'frame') {
      if (latestFrame) {
        // 上一帧还没渲染就被新帧顶替: 直接归还旧 buffer (丢帧保延迟)
        worker.postMessage({ type: 'recycle', buffer: latestFrame.buffer }, [latestFrame.buffer]);
      }
      latestFrame = msg;
    }
  };
  worker.postMessage({ type: 'init', N: res });
  pushParams();
}

function pushParams() {
  worker.postMessage({
    type: 'params',
    params: {
      visc: parseFloat(ui.viscosity.value),
      diff: parseFloat(ui.diffusion.value),
      vorticity: ui.vorticity.checked ? 2.0 : 0,
      fade: 0.15,
    },
  });
}

// ---- 交互: 指针拖动 -> 批量 splat ----
let pointerDown = false;
let lastPos = null;
const splatQueue = [];

function canvasToGrid(ev) {
  const r = canvas.getBoundingClientRect();
  return {
    x: ((ev.clientX - r.left) / r.width) * N + 0.5,
    y: ((ev.clientY - r.top) / r.height) * N + 0.5,
  };
}

canvas.addEventListener('pointerdown', (ev) => {
  pointerDown = true;
  canvas.setPointerCapture(ev.pointerId);
  lastPos = canvasToGrid(ev);
});
canvas.addEventListener('pointerup', () => { pointerDown = false; lastPos = null; });
canvas.addEventListener('pointermove', (ev) => {
  if (!pointerDown || !lastPos) return;
  const pos = canvasToGrid(ev);
  const dx = pos.x - lastPos.x;
  const dy = pos.y - lastPos.y;
  const force = parseFloat(ui.force.value);
  const dye = parseFloat(ui.dye.value);
  // 沿路径插值, 快速挥动也不会断线
  const dist = Math.hypot(dx, dy);
  const steps = Math.max(1, Math.min(8, Math.ceil(dist / 2)));
  for (let s = 0; s < steps; s++) {
    const t = s / steps;
    splatQueue.push({
      x: lastPos.x + dx * t,
      y: lastPos.y + dy * t,
      radius: Math.max(2, N / 32),
      fx: dx * force,
      fy: dy * force,
      dye,
    });
  }
  lastPos = pos;
});

// 每帧把队列里的 splat 合并成一条消息发给 worker
function flushSplats() {
  if (splatQueue.length === 0) return;
  for (const s of splatQueue) worker.postMessage({ type: 'splat', splat: s });
  splatQueue.length = 0;
}

// ---- 自动降级 ----
let lowFpsSince = 0;
function autoDegrade(now) {
  if (!ui.autoDegrade.checked) return;
  const idx = RESOLUTIONS.indexOf(N);
  if (fpsEMA > 0 && fpsEMA < 45 && idx > 0 && RESOLUTIONS[idx - 1] >= MIN_AUTO_RES) {
    if (lowFpsSince === 0) lowFpsSince = now;
    if (now - lowFpsSince > 2000) {
      const next = RESOLUTIONS[idx - 1];
      degradedNotice = `已自动降级 ${N}→${next} (FPS ${fpsEMA.toFixed(0)})`;
      ui.degraded.textContent = degradedNotice;
      setResolution(next, true);
      lowFpsSince = 0;
    }
  } else {
    lowFpsSince = 0;
  }
}

function setResolution(res, fromAuto = false) {
  N = res;
  ui.resolution.value = String(res);
  if (!fromAuto) degradedNotice = '';
  ui.degraded.textContent = degradedNotice;
  spawnWorker(res);
}

// ---- 主循环 ----
function frame(now) {
  const dt = now - lastRAF;
  lastRAF = now;
  if (dt > 0 && dt < 1000) {
    const fps = 1000 / dt;
    fpsEMA = fpsEMA === 0 ? fps : fpsEMA * 0.95 + fps * 0.05;
  }

  flushSplats();

  if (latestFrame) {
    const msg = latestFrame;
    latestFrame = null;
    renderer.draw(new Uint8Array(msg.buffer), msg.N);
    worker.postMessage({ type: 'recycle', buffer: msg.buffer }, [msg.buffer]);
    ui.simMs.textContent = msg.simMs.toFixed(2) + ' ms';
    ui.steps.textContent = String(msg.stepCount);
  }

  ui.fps.textContent = fpsEMA.toFixed(0);
  autoDegrade(now);
  requestAnimationFrame(frame);
}

// ---- UI 事件 ----
ui.resolution.addEventListener('change', () => setResolution(parseInt(ui.resolution.value, 10)));
for (const el of [ui.viscosity, ui.diffusion, ui.vorticity]) {
  el.addEventListener('input', pushParams);
}
ui.pause.addEventListener('click', () => {
  paused = !paused;
  worker.postMessage({ type: paused ? 'pause' : 'resume' });
  ui.pause.textContent = paused ? '继续' : '暂停';
});
ui.clear.addEventListener('click', () => worker.postMessage({ type: 'clear' }));

function resizeCanvas() {
  const side = Math.min(window.innerWidth - 340, window.innerHeight - 40);
  const s = Math.max(300, Math.min(900, side));
  canvas.width = s;
  canvas.height = s;
}
window.addEventListener('resize', resizeCanvas);

resizeCanvas();
spawnWorker(N);
requestAnimationFrame(frame);
