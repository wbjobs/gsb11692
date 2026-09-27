import { FluidGL } from './fluid-gl.js';

const $ = (id) => document.getElementById(id);
const glCanvas = $('glCanvas');
const cpuCanvas = $('cpuCanvas');
const cpuCtx = cpuCanvas.getContext('2d');
const stage = $('stage');
const overlay = $('overlay');

const RESOLUTIONS = [64, 128, 256, 512, 1024];
const CPU_MAX_RES = 512;

const state = {
  mode: 'webgl',
  res: 128,
  iters: 30,
  force: 1.2,
  radius: 0.0018,
  dyeDiss: 0.35,
  velDiss: 0.10,
  autoDegrade: true,
  paused: false,
};

let glSim = null;
let worker = null;
let cpuPending = false;          // 上一帧 worker 尚未返回 -> 不再堆叠请求, 控制交互延迟
const splatQueue = [];           // CPU 模式下待发送的喷溅
let degradeCount = 0;
let lastDegradeAt = 0;
let slowFrames = 0;

const perf = { longtasks: 0, longtaskMs: 0, simMs: 0, frameMs: 16.7, simFrames: 0, simGapMs: 16.7, fps: 0 };

// ---------- 日志 / 覆盖层 ----------
function log(msg, warn = false) {
  const li = document.createElement('li');
  li.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  if (warn) li.className = 'warn';
  const list = $('log');
  list.prepend(li);
  while (list.children.length > 8) list.lastChild.remove();
}
function setOverlay(text) {
  if (text) { overlay.textContent = text; overlay.classList.remove('hidden'); }
  else overlay.classList.add('hidden');
}

// ---------- PerformanceObserver ----------
if ('PerformanceObserver' in window) {
  const supported = PerformanceObserver.supportedEntryTypes || [];
  if (supported.includes('longtask')) {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) {
        perf.longtasks++;
        perf.longtaskMs += e.duration;
      }
    }).observe({ entryTypes: ['longtask'] });
  }
  if (supported.includes('measure')) {
    new PerformanceObserver((list) => {
      const entries = list.getEntries();
      if (entries.length) perf.simMs = entries[entries.length - 1].duration;
    }).observe({ entryTypes: ['measure'] });
  }
}

// ---------- 求解器 ----------
function initGL() {
  try {
    glSim = new FluidGL(glCanvas, state.res);
    return true;
  } catch (err) {
    log(`WebGL 初始化失败: ${err.message}, 回退 CPU 模式`, true);
    glSim = null;
    return false;
  }
}

function initWorker() {
  if (!worker) {
    worker = new Worker('js/fluid-worker.js');
    worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.cmd === 'frame') {
        cpuPending = false;
        perf.simMs = msg.simMs;
        perf.simFrames++;
        const now = performance.now();
        if (perf.lastCpuFrameAt) perf.simGapMs = perf.simGapMs * 0.9 + (now - perf.lastCpuFrameAt) * 0.1;
        perf.lastCpuFrameAt = now;
        if (cpuCanvas.width !== msg.n) { cpuCanvas.width = msg.n; cpuCanvas.height = msg.n; }
        cpuCtx.putImageData(new ImageData(new Uint8ClampedArray(msg.buffer), msg.n, msg.n), 0, 0);
      }
    };
    worker.onerror = (err) => log(`Worker 错误: ${err.message}`, true);
  }
  cpuPending = false;
  worker.postMessage({ cmd: 'init', n: Math.min(state.res, CPU_MAX_RES) });
}

function cpuMemoryBytes(n) {
  return 12 * (n + 2) * (n + 2) * 4 + n * n * 4; // 12 个标量场 + RGBA 帧缓冲
}

// ---------- 分辨率 / 模式切换 ----------
function applyResolution(res, reason = '') {
  if (state.mode === 'cpu' && res > CPU_MAX_RES) {
    res = CPU_MAX_RES;
    reason = reason || `CPU 模式分辨率上限 ${CPU_MAX_RES}, 已降低`;
  }
  state.res = res;
  $('resolution').value = String(res);
  if (state.mode === 'webgl' && glSim) glSim.setResolution(res);
  if (worker) worker.postMessage({ cmd: 'init', n: Math.min(res, CPU_MAX_RES) });
  updateStats();
  if (reason) log(reason);
}

function applyMode(mode) {
  state.mode = mode;
  $('mode').value = mode;
  if (mode === 'webgl') {
    if (!glSim && !initGL()) { applyMode('cpu'); return; }
    glCanvas.hidden = false;
    cpuCanvas.hidden = true;
    setOverlay('WebGL2 (GPU) 模式');
    setTimeout(() => setOverlay(''), 1500);
  } else {
    if (state.res > CPU_MAX_RES) applyResolution(CPU_MAX_RES, 'CPU 模式分辨率上限 512, 已降低');
    initWorker();
    glCanvas.hidden = true;
    cpuCanvas.hidden = false;
    setOverlay('CPU (Web Worker) 模式');
    setTimeout(() => setOverlay(''), 1500);
  }
  updateStats();
}

glCanvas.addEventListener('webglcontextlost', (e) => {
  e.preventDefault();
  glSim = null;
  log('WebGL 上下文丢失, 回退 CPU 模式', true);
  applyMode('cpu');
});

// ---------- 自动降级 ----------
function maybeDegrade() {
  if (!state.autoDegrade || state.paused) return;
  // CPU 模式下主线程 rAF 不被 worker 阻塞, 需用 worker 帧到达间隔衡量真实帧率
  const effMs = state.mode === 'cpu' ? perf.simGapMs : perf.frameMs;
  if (effMs > 30) slowFrames++; else slowFrames = 0;
  const now = performance.now();
  if (slowFrames < 60 || now - lastDegradeAt < 2000) return;
  lastDegradeAt = now;
  slowFrames = 0;
  degradeCount++;

  const idx = RESOLUTIONS.indexOf(state.res);
  if (idx > 0) {
    const next = RESOLUTIONS[idx - 1];
    applyResolution(next, `帧耗时 ${effMs.toFixed(1)}ms 过高, 自动降级: ${state.res} → ${next}`);
  } else if (state.iters > 8) {
    state.iters = Math.max(8, state.iters >> 1);
    $('iters').value = state.iters;
    $('itersVal').textContent = state.iters;
    log(`已达最低分辨率, 压力迭代降至 ${state.iters}`, true);
  } else {
    log('已达最低画质, 无法继续降级', true);
  }
}

// ---------- 鼠标交互 ----------
const pointer = { x: 0.5, y: 0.5, px: 0.5, py: 0.5, moved: false, down: false };
let hue = Math.random();

function hsv2rgb(h, s, v) {
  const i = Math.floor(h * 6), f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
}
function nextColor() {
  hue = (hue + 0.013) % 1;
  return hsv2rgb(hue, 1, 1).map((c) => c * 0.35);
}

function updatePointer(e) {
  const rect = stage.getBoundingClientRect();
  pointer.px = pointer.x;
  pointer.py = pointer.y;
  pointer.x = (e.clientX - rect.left) / rect.width;
  pointer.y = 1 - (e.clientY - rect.top) / rect.height; // uv 原点在左下
  pointer.moved = true;
}
stage.addEventListener('pointermove', updatePointer);
stage.addEventListener('pointerdown', (e) => { pointer.down = true; updatePointer(e); });
addEventListener('pointerup', () => { pointer.down = false; });

function emitPointerSplat(dt) {
  if (!pointer.moved) return;
  pointer.moved = false;
  let dx = ((pointer.x - pointer.px) / Math.max(dt, 1e-4)) * state.force * 0.15;
  let dy = ((pointer.y - pointer.py) / Math.max(dt, 1e-4)) * state.force * 0.15;
  const cap = 4; // 限幅, 防止极端输入导致数值爆炸
  dx = Math.max(-cap, Math.min(cap, dx));
  dy = Math.max(-cap, Math.min(cap, dy));
  const boost = pointer.down ? 1.8 : 1.0;
  const color = nextColor().map((c) => c * boost);
  if (state.mode === 'webgl' && glSim) {
    const aspect = glCanvas.width / glCanvas.height;
    glSim.splat(pointer.x, pointer.y, dx, dy, color, state.radius, aspect);
  } else {
    splatQueue.push({ x: pointer.x, y: pointer.y, dx, dy,
                      r: color[0], g: color[1], b: color[2], radius: state.radius });
  }
}

// ---------- 主循环 ----------
let lastT = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const rawDt = (now - lastT) / 1000;
  lastT = now;
  const dt = Math.min(rawDt, 1 / 30); // 钳制 dt, 保证半拉格朗日回溯稳定
  perf.frameMs = perf.frameMs * 0.92 + rawDt * 1000 * 0.08;
  if (state.paused) return;

  performance.mark('sim-start');
  emitPointerSplat(dt);
  if (state.mode === 'webgl' && glSim) {
    glSim.step(dt, state);
    glSim.render();
    perf.simFrames++;
  } else if (worker && !cpuPending) {
    cpuPending = true;
    worker.postMessage({ cmd: 'frame', dt, params: state, splats: splatQueue.splice(0) });
  }
  performance.mark('sim-end');
  performance.measure('fluid-sim', 'sim-start', 'sim-end');
  maybeDegrade();
}

// ---------- 统计 ----------
function updateStats() {
  $('stFps').textContent = perf.fps.toFixed(0);
  $('stFrame').textContent = perf.frameMs.toFixed(1) + ' ms';
  $('stSim').textContent = perf.simMs.toFixed(2) + ' ms';
  const n = state.mode === 'cpu' ? Math.min(state.res, CPU_MAX_RES) : state.res;
  $('stGrid').textContent = `${n} × ${n} (${(n * n).toLocaleString()} 格)`;
  const bytes = state.mode === 'webgl' && glSim ? glSim.memoryBytes() : cpuMemoryBytes(n);
  $('stMem').textContent = (bytes / 1048576).toFixed(1) + ' MB';
  $('stLongtask').textContent = `${perf.longtasks} (${perf.longtaskMs.toFixed(0)} ms)`;
  $('stDegrade').textContent = degradeCount;
}
setInterval(() => {
  perf.fps = perf.simFrames / 0.25; // 实际完成的模拟帧率
  perf.simFrames = 0;
  performance.clearMeasures('fluid-sim'); // 防止 measure 缓冲无限增长
  performance.clearMarks('sim-start');
  performance.clearMarks('sim-end');
  updateStats();
}, 250);

// ---------- 控件 ----------
function bindRange(id, key, fmt = (v) => v) {
  const el = $(id);
  el.addEventListener('input', () => {
    state[key] = parseFloat(el.value);
    $(id + 'Val').textContent = fmt(state[key]);
  });
}
bindRange('iters', 'iters', (v) => v.toFixed(0));
bindRange('force', 'force', (v) => v.toFixed(1));
bindRange('radius', 'radius', (v) => v.toFixed(4));
bindRange('dyeDiss', 'dyeDiss', (v) => v.toFixed(2));
bindRange('velDiss', 'velDiss', (v) => v.toFixed(2));

$('mode').addEventListener('change', (e) => applyMode(e.target.value));
$('resolution').addEventListener('change', (e) => applyResolution(parseInt(e.target.value, 10)));
$('autoDegrade').addEventListener('change', (e) => { state.autoDegrade = e.target.checked; });
$('pauseBtn').addEventListener('click', () => {
  state.paused = !state.paused;
  $('pauseBtn').textContent = state.paused ? '继续' : '暂停';
});
$('resetBtn').addEventListener('click', () => {
  if (glSim) glSim.clear();
  if (worker) worker.postMessage({ cmd: 'init', n: Math.min(state.res, CPU_MAX_RES) });
  log('已重置流场');
});

// ---------- 画布尺寸 ----------
function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const rect = stage.getBoundingClientRect();
  glCanvas.width = Math.max(1, Math.round(rect.width * dpr));
  glCanvas.height = Math.max(1, Math.round(rect.height * dpr));
}
new ResizeObserver(resize).observe(stage);
resize();

// ---------- 启动 ----------
if (!initGL()) applyMode('cpu');
else {
  // 初始几团烟雾
  for (let i = 0; i < 5; i++) {
    const x = 0.2 + 0.6 * Math.random(), y = 0.2 + 0.6 * Math.random();
    const a = Math.random() * Math.PI * 2;
    glSim.splat(x, y, Math.cos(a) * 1.5, Math.sin(a) * 1.5, nextColor(), 0.002, glCanvas.width / glCanvas.height);
  }
  setOverlay('');
  log('WebGL2 求解器已启动');
}
initWorker(); // 预创建, 模式切换无延迟
updateStats();
requestAnimationFrame(frame);
