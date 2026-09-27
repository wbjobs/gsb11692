// CPU Stable-Fluids 求解器 (Jos Stam), 运行在 Web Worker 中
// 网格 (N+2)x(N+2), Float32Array; 边界条件通过 setBnd 显式处理
// 每帧回传 Uint8ClampedArray (transferable), 主线程零拷贝渲染

let N = 0;
let u, v, u0, v0, p, div;           // 速度 / 压力 / 散度
let r, g, b, r0, g0, b0;            // 染料 RGB

function IX(i, j) { return i + (N + 2) * j; }

function alloc(n) {
  N = n;
  const size = (N + 2) * (N + 2);
  u = new Float32Array(size);  v = new Float32Array(size);
  u0 = new Float32Array(size); v0 = new Float32Array(size);
  p = new Float32Array(size);  div = new Float32Array(size);
  r = new Float32Array(size);  g = new Float32Array(size);  b = new Float32Array(size);
  r0 = new Float32Array(size); g0 = new Float32Array(size); b0 = new Float32Array(size);
}

// b=1: x 方向速度无滑移; b=2: y 方向; b=0: 标量 Neumann
function setBnd(bnd, x) {
  for (let i = 1; i <= N; i++) {
    x[IX(0, i)]     = bnd === 1 ? -x[IX(1, i)] : x[IX(1, i)];
    x[IX(N + 1, i)] = bnd === 1 ? -x[IX(N, i)] : x[IX(N, i)];
    x[IX(i, 0)]     = bnd === 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
    x[IX(i, N + 1)] = bnd === 2 ? -x[IX(i, N)] : x[IX(i, N)];
  }
  x[IX(0, 0)]         = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
  x[IX(0, N + 1)]     = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
  x[IX(N + 1, 0)]     = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
  x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

// 投影: 构造无散速度场 (压力泊松方程, Jacobi 迭代)
function project(iters) {
  const h = 1.0 / N;
  for (let j = 1; j <= N; j++) {
    for (let i = 1; i <= N; i++) {
      div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
                                  v[IX(i, j + 1)] - v[IX(i, j - 1)]);
      p[IX(i, j)] *= 0.8; // 压力衰减, 防止误差累积
    }
  }
  setBnd(0, div); setBnd(0, p);
  for (let k = 0; k < iters; k++) {
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                                       p[IX(i, j - 1)] + p[IX(i, j + 1)]) * 0.25;
      }
    }
    setBnd(0, p);
  }
  for (let j = 1; j <= N; j++) {
    for (let i = 1; i <= N; i++) {
      u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
      v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
    }
  }
  setBnd(1, u); setBnd(2, v);
}

// 半拉格朗日平流 + 双线性插值, 无条件稳定
function advect(bnd, d, d0, dt) {
  const dt0 = dt * N;
  for (let j = 1; j <= N; j++) {
    for (let i = 1; i <= N; i++) {
      let x = i - dt0 * u[IX(i, j)];
      let y = j - dt0 * v[IX(i, j)];
      if (x < 0.5) x = 0.5; if (x > N + 0.5) x = N + 0.5;
      if (y < 0.5) y = 0.5; if (y > N + 0.5) y = N + 0.5;
      const i0 = x | 0, i1 = i0 + 1;
      const j0 = y | 0, j1 = j0 + 1;
      const s1 = x - i0, s0 = 1 - s1;
      const t1 = y - j0, t0 = 1 - t1;
      d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                    s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
  }
  setBnd(bnd, d);
}

function applySplat(s) {
  // s.x, s.y: uv 坐标 [0,1]; s.dx, s.dy: uv/s; radius: uv 单位
  const gx = 0.5 + s.x * N, gy = 0.5 + s.y * N;
  const gr = Math.max(1, s.radius * N);
  const reach = Math.ceil(gr * 3);
  const iMin = Math.max(1, Math.floor(gx - reach)), iMax = Math.min(N, Math.ceil(gx + reach));
  const jMin = Math.max(1, Math.floor(gy - reach)), jMax = Math.min(N, Math.ceil(gy + reach));
  const invR2 = 1 / (gr * gr);
  for (let j = jMin; j <= jMax; j++) {
    for (let i = iMin; i <= iMax; i++) {
      const dx = i - gx, dy = j - gy;
      const fall = Math.exp(-(dx * dx + dy * dy) * invR2);
      const idx = IX(i, j);
      u[idx] += s.dx * N * fall;   // uv/s -> 格/s
      v[idx] += s.dy * N * fall;
      r[idx] += s.r * fall;
      g[idx] += s.g * fall;
      b[idx] += s.b * fall;
    }
  }
}

function step(dt, params, splats) {
  for (const s of splats) applySplat(s);

  // 速度自平流 -> 投影
  u0.set(u); v0.set(v);
  advect(1, u, u0, dt);
  advect(2, v, v0, dt);
  const velDecay = 1 / (1 + params.velDiss * dt);
  for (let i = 0; i < u.length; i++) { u[i] *= velDecay; v[i] *= velDecay; }
  project(params.iters);

  // 染料平流
  r0.set(r); g0.set(g); b0.set(b);
  advect(0, r, r0, dt);
  advect(0, g, g0, dt);
  advect(0, b, b0, dt);
  const dyeDecay = 1 / (1 + params.dyeDiss * dt);
  for (let i = 0; i < r.length; i++) { r[i] *= dyeDecay; g[i] *= dyeDecay; b[i] *= dyeDecay; }
}

function renderFrame() {
  const out = new Uint8ClampedArray(N * N * 4);
  for (let j = 1; j <= N; j++) {
    const row = (N - j) * N; // 翻转 y: 网格行 1 在底部
    for (let i = 1; i <= N; i++) {
      const idx = IX(i, j);
      const o = (row + i - 1) * 4;
      const cr = r[idx], cg = g[idx], cb = b[idx];
      const m = 1 / (1 + 0.15 * Math.max(cr, cg, cb)); // 与 GPU 端一致的色调映射
      out[o]     = cr * m * 255;
      out[o + 1] = cg * m * 255;
      out[o + 2] = cb * m * 255;
      out[o + 3] = 255;
    }
  }
  return out;
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.cmd === 'init') {
    alloc(msg.n);
    self.postMessage({ cmd: 'ready', n: N });
    return;
  }
  if (msg.cmd === 'frame') {
    if (!N) return;
    const t0 = performance.now();
    step(msg.dt, msg.params, msg.splats);
    const frame = renderFrame();
    const simMs = performance.now() - t0;
    self.postMessage({ cmd: 'frame', buffer: frame.buffer, n: N, simMs }, [frame.buffer]);
  }
};
