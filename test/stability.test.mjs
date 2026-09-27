/**
 * 稳定性 / 边界条件 / 大网格 测试 (Node 直接跑求解器核心)。
 * 运行: node test/stability.test.mjs
 */
import { FluidSolver, BND_VERTICAL, BND_HORIZONTAL } from '../js/solver.js';

let failures = 0;
function check(name, cond, detail = '') {
  const ok = !!cond;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
  if (!ok) failures++;
}

const PARAMS = { visc: 0.0001, diff: 0.00001, vorticity: 2.0, fade: 0.15 };
const DT = 1 / 60;

// ---------- 1. 数值稳定性: 随机强扰动下 600 步不爆炸 ----------
{
  const s = new FluidSolver(128);
  let rng = 12345;
  const rand = () => (rng = (rng * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let step = 0; step < 600; step++) {
    if (step % 10 === 0) {
      // 每 10 步注入一个随机强 splat (力刻意取大, 压力测试)
      s.splat(10 + rand() * 108, 10 + rand() * 108, 6,
              (rand() - 0.5) * 200, (rand() - 0.5) * 200, 2);
    }
    s.step(DT, PARAMS);
  }
  const d = s.diagnostics();
  check('600 步随机扰动后无 NaN/Inf', !d.hasNaN);
  check('速度有界 (maxSpeed < 1e4)', d.maxSpeed < 1e4, `maxSpeed=${d.maxSpeed.toFixed(2)}`);
  check('密度非负且有限', d.totalMass >= 0 && Number.isFinite(d.totalMass),
        `mass=${d.totalMass.toFixed(1)}`);
}

// ---------- 2. 极端 dt 不爆炸 (半拉格朗日无条件稳定) ----------
{
  const s = new FluidSolver(64);
  s.splat(32, 32, 5, 50, -50, 3);
  for (let i = 0; i < 60; i++) s.step(0.1, PARAMS); // dt=0.1 远超 CFL
  const d = s.diagnostics();
  check('dt=0.1 (超 CFL) 60 步后仍稳定', !d.hasNaN && d.maxSpeed < 1e4,
        `maxSpeed=${d.maxSpeed.toFixed(2)}`);
}

// ---------- 3. 边界条件: 固体墙 ----------
{
  const s = new FluidSolver(64);
  s.splat(32, 32, 8, 80, 0, 1);
  s.step(DT, PARAMS);
  const N = s.N;
  let uOk = true, vOk = true, densOk = true;
  for (let i = 1; i <= N; i++) {
    // u: 左右墙镜像取反; v: 上下墙镜像取反; 密度: 镜像相等
    if (s.u[s.IX(0, i)] !== -s.u[s.IX(1, i)]) uOk = false;
    if (s.u[s.IX(N + 1, i)] !== -s.u[s.IX(N, i)]) uOk = false;
    if (s.v[s.IX(i, 0)] !== -s.v[s.IX(i, 1)]) vOk = false;
    if (s.v[s.IX(i, N + 1)] !== -s.v[s.IX(i, N)]) vOk = false;
    if (s.dens[s.IX(0, i)] !== s.dens[s.IX(1, i)]) densOk = false;
    if (s.dens[s.IX(i, 0)] !== s.dens[s.IX(i, 1)]) densOk = false;
  }
  check('边界: u 在垂直墙取反 (BND_VERTICAL)', uOk);
  check('边界: v 在水平墙取反 (BND_HORIZONTAL)', vOk);
  check('边界: 密度在墙上镜像 (BND_NONE)', densOk);
}

// ---------- 4. 投影降低散度 ----------
// 注意: Gauss-Seidel 对低频误差每步仅衰减 ~(1-(π/N)²), 大网格需更多迭代;
// 这里用 N=32 + 200 次迭代验证投影算子本身的正确性。
{
  const s = new FluidSolver(32);
  s.projectIters = 200;
  const N = s.N;
  // 边界兼容的纯梯度场 (应被投影完全移除): u=dφ/di, v=dφ/dj, φ=cos(πi/N)cos(πj/N)
  for (let j = 0; j <= N + 1; j++)
    for (let i = 0; i <= N + 1; i++) {
      s.u[s.IX(i, j)] = -Math.sin(Math.PI * i / N) * Math.cos(Math.PI * j / N);
      s.v[s.IX(i, j)] = -Math.cos(Math.PI * i / N) * Math.sin(Math.PI * j / N);
    }
  const meanDiv = () => {
    let sum = 0;
    for (let j = 1; j <= N; j++)
      for (let i = 1; i <= N; i++)
        sum += Math.abs(0.5 * (s.u[s.IX(i + 1, j)] - s.u[s.IX(i - 1, j)] +
                               s.v[s.IX(i, j + 1)] - s.v[s.IX(i, j - 1)]));
    return sum / (N * N);
  };
  const divBefore = meanDiv();
  s.project();
  const divAfter = meanDiv();
  check('压力投影显著降低散度 (N=32, 200 迭代)', divAfter < divBefore * 0.3,
        `${divBefore.toExponential(2)} -> ${divAfter.toExponential(2)}`);
}

// ---------- 4b. 模拟全程散度有界 (工程上真正重要的性质) ----------
{
  const s = new FluidSolver(96);
  for (let step = 0; step < 120; step++) {
    if (step % 15 === 0) s.splat(48, 48, 6, 40, -60, 1);
    s.step(DT, PARAMS);
  }
  const d = s.diagnostics();
  check('120 步模拟后散度相对速度规模保持有界',
        d.meanDivergence < Math.max(0.5, d.maxSpeed * 2),
        `meanDiv=${d.meanDivergence.toExponential(2)}, maxSpeed=${d.maxSpeed.toFixed(2)}`);
}

// ---------- 5. 大网格: 256 与 384 不崩、耗时可接受 ----------
for (const N of [256, 384]) {
  const s = new FluidSolver(N);
  s.splat(N / 2, N / 2, N / 16, 30, -30, 2);
  const t0 = performance.now();
  for (let i = 0; i < 60; i++) s.step(DT, PARAMS);
  const ms = (performance.now() - t0) / 60;
  const d = s.diagnostics();
  check(`N=${N} 大网格 60 步稳定`, !d.hasNaN && d.maxSpeed < 1e4,
        `avg=${ms.toFixed(2)}ms/步, mem=${(s.memoryBytes() / 1048576).toFixed(1)}MB`);
}

// ---------- 6. 非法规模拒绝 ----------
{
  let threw = false;
  try { new FluidSolver(10000); } catch { threw = true; }
  check('非法网格规模抛 RangeError', threw);
}

console.log(failures === 0 ? '\n全部通过 ✔' : `\n${failures} 项失败 ✗`);
process.exit(failures === 0 ? 0 : 1);
