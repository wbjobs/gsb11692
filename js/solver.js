/**
 * solver.js — 基于 Jos Stam "Stable Fluids" 的网格法 Navier-Stokes 简化求解器。
 *
 * 数值要点:
 *  - 半拉格朗日平流 (无条件稳定, dt 大也不爆炸)
 *  - 粘性/扩散用 Gauss-Seidel 隐式迭代 (稳定)
 *  - Helmholtz-Hodge 投影保证无散度 (不可压缩)
 *  - 涡度限制 (vorticity confinement) 补偿数值耗散, 让烟雾更"卷"
 *  - 全部状态放在 TypedArray, 双缓冲, 零 GC 压力
 *  - 边界条件: 固体墙 (法向速度为 0, 切向镜像取反)
 *
 * 该文件是纯 ES module, 同时被 Web Worker 和 Node 测试引用。
 */

export const BND_NONE = 0;
export const BND_VERTICAL = 1;   // u 分量
export const BND_HORIZONTAL = 2; // v 分量

export class FluidSolver {
  /**
   * @param {number} N 内部网格边长 (不含边界), 总数组 (N+2)^2
   */
  constructor(N) {
    if (!Number.isInteger(N) || N < 16 || N > 512) {
      throw new RangeError(`非法网格规模 N=${N} (允许 16..512)`);
    }
    this.N = N;
    this.size = (N + 2) * (N + 2);
    // 双缓冲速度场 / 密度场 + 投影与涡度用的临时场
    this.u = new Float32Array(this.size);
    this.v = new Float32Array(this.size);
    this.uPrev = new Float32Array(this.size);
    this.vPrev = new Float32Array(this.size);
    this.dens = new Float32Array(this.size);
    this.densPrev = new Float32Array(this.size);
    this.pressure = new Float32Array(this.size);
    this.divergence = new Float32Array(this.size);
    this.curl = new Float32Array(this.size);
    // 迭代次数随分辨率缩放, 大网格不增加 => 内存/耗时可控
    this.projectIters = 20;
    this.diffuseIters = 4;
  }

  /** 估算求解器常驻内存 (字节) */
  memoryBytes() {
    return this.size * 4 * 9;
  }

  reset() {
    for (const a of [this.u, this.v, this.uPrev, this.vPrev,
                     this.dens, this.densPrev, this.pressure,
                     this.divergence, this.curl]) a.fill(0);
  }

  IX(i, j) { return i + (this.N + 2) * j; }

  /**
   * 边界条件: 四周为固体墙。
   * b=1: 垂直边速度取反; b=2: 水平边速度取反; b=0: 标量(密度/压力)镜像。
   * 角点取两邻点平均, 避免奇异。
   */
  setBnd(b, x) {
    const N = this.N;
    for (let i = 1; i <= N; i++) {
      x[this.IX(0, i)]     = b === BND_VERTICAL   ? -x[this.IX(1, i)] : x[this.IX(1, i)];
      x[this.IX(N + 1, i)] = b === BND_VERTICAL   ? -x[this.IX(N, i)] : x[this.IX(N, i)];
      x[this.IX(i, 0)]     = b === BND_HORIZONTAL ? -x[this.IX(i, 1)] : x[this.IX(i, 1)];
      x[this.IX(i, N + 1)] = b === BND_HORIZONTAL ? -x[this.IX(i, N)] : x[this.IX(i, N)];
    }
    x[this.IX(0, 0)]         = 0.5 * (x[this.IX(1, 0)]     + x[this.IX(0, 1)]);
    x[this.IX(0, N + 1)]     = 0.5 * (x[this.IX(1, N + 1)] + x[this.IX(0, N)]);
    x[this.IX(N + 1, 0)]     = 0.5 * (x[this.IX(N, 0)]     + x[this.IX(N + 1, 1)]);
    x[this.IX(N + 1, N + 1)] = 0.5 * (x[this.IX(N, N + 1)] + x[this.IX(N + 1, N)]);
  }

  /** Gauss-Seidel 隐式扩散: (I - dt*diff*Lap) x = x0 */
  linSolve(b, x, x0, a, c, iters) {
    const N = this.N;
    const invC = 1 / c;
    for (let k = 0; k < iters; k++) {
      for (let j = 1; j <= N; j++) {
        let row = this.IX(1, j);
        for (let i = 1; i <= N; i++, row++) {
          x[row] = (x0[row] + a * (x[row - 1] + x[row + 1] +
                                   x[row - (N + 2)] + x[row + (N + 2)])) * invC;
        }
      }
      this.setBnd(b, x);
    }
  }

  diffuse(b, x, x0, diff, dt) {
    const a = dt * diff * this.N * this.N;
    this.linSolve(b, x, x0, a, 1 + 4 * a, this.diffuseIters);
  }

  /** Helmholtz-Hodge 投影: 减去压力梯度使速度场无散 */
  project() {
    const N = this.N;
    const h = 1 / N;
    const { u, v, pressure: p, divergence: div } = this;
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        const idx = this.IX(i, j);
        div[idx] = -0.5 * h * (u[this.IX(i + 1, j)] - u[this.IX(i - 1, j)] +
                               v[this.IX(i, j + 1)] - v[this.IX(i, j - 1)]);
        p[idx] = 0;
      }
    }
    this.setBnd(BND_NONE, div);
    this.setBnd(BND_NONE, p);
    this.linSolve(BND_NONE, p, div, 1, 4, this.projectIters);
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        const idx = this.IX(i, j);
        u[idx] -= 0.5 * (p[this.IX(i + 1, j)] - p[this.IX(i - 1, j)]) / h;
        v[idx] -= 0.5 * (p[this.IX(i, j + 1)] - p[this.IX(i, j - 1)]) / h;
      }
    }
    this.setBnd(BND_VERTICAL, u);
    this.setBnd(BND_HORIZONTAL, v);
  }

  /** 半拉格朗日平流: 回溯粒子轨迹 + 双线性插值 */
  advect(b, d, d0, u, v, dt) {
    const N = this.N;
    const dt0 = dt * N;
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        const idx = this.IX(i, j);
        let x = i - dt0 * u[idx];
        let y = j - dt0 * v[idx];
        // 钳制到网格内, 防止回溯出界
        if (x < 0.5) x = 0.5; else if (x > N + 0.5) x = N + 0.5;
        if (y < 0.5) y = 0.5; else if (y > N + 0.5) y = N + 0.5;
        const i0 = x | 0, j0 = y | 0;
        const i1 = i0 + 1, j1 = j0 + 1;
        const s1 = x - i0, s0 = 1 - s1;
        const t1 = y - j0, t0 = 1 - t1;
        d[idx] = s0 * (t0 * d0[this.IX(i0, j0)] + t1 * d0[this.IX(i0, j1)]) +
                 s1 * (t0 * d0[this.IX(i1, j0)] + t1 * d0[this.IX(i1, j1)]);
      }
    }
    this.setBnd(b, d);
  }

  /** 涡度限制力: 沿 ∇|ω| × ω 方向加力, 恢复被数值耗散抹掉的小尺度卷曲 */
  vorticityConfinement(dt, strength) {
    const N = this.N;
    const { u, v, curl } = this;
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        curl[this.IX(i, j)] = 0.5 * ((v[this.IX(i + 1, j)] - v[this.IX(i - 1, j)]) -
                                     (u[this.IX(i, j + 1)] - u[this.IX(i, j - 1)]));
      }
    }
    for (let j = 2; j < N; j++) {
      for (let i = 2; i < N; i++) {
        const idx = this.IX(i, j);
        const dwx = 0.5 * (Math.abs(curl[this.IX(i + 1, j)]) - Math.abs(curl[this.IX(i - 1, j)]));
        const dwy = 0.5 * (Math.abs(curl[this.IX(i, j + 1)]) - Math.abs(curl[this.IX(i, j - 1)]));
        const len = Math.sqrt(dwx * dwx + dwy * dwy) + 1e-5;
        const c = curl[idx];
        u[idx] += dt * strength * (dwy / len) * -c;
        v[idx] += dt * strength * (dwx / len) * c;
      }
    }
  }

  /** 添加力 / 染料源 (交互入口, 网格坐标) */
  splat(cx, cy, radius, fx, fy, dyeAmount) {
    const N = this.N;
    const r2 = radius * radius;
    const iMin = Math.max(1, Math.floor(cx - radius));
    const iMax = Math.min(N, Math.ceil(cx + radius));
    const jMin = Math.max(1, Math.floor(cy - radius));
    const jMax = Math.min(N, Math.ceil(cy + radius));
    for (let j = jMin; j <= jMax; j++) {
      for (let i = iMin; i <= iMax; i++) {
        const dx = i - cx, dy = j - cy;
        const d2 = dx * dx + dy * dy;
        if (d2 > r2) continue;
        const fall = Math.exp(-d2 / (r2 * 0.5));
        const idx = this.IX(i, j);
        this.u[idx] += fx * fall;
        this.v[idx] += fy * fall;
        this.dens[idx] = Math.min(this.dens[idx] + dyeAmount * fall, 8);
      }
    }
  }

  /**
   * 单步推进。dt 由调用方钳制 (建议 <= 1/30), 内部无条件稳定。
   * @param {{visc:number, diff:number, vorticity:number, fade:number}} p
   */
  step(dt, p) {
    const { u, v, uPrev, vPrev, dens, densPrev } = this;

    if (p.vorticity > 0) this.vorticityConfinement(dt, p.vorticity);

    // 速度: 扩散 -> 投影 -> 平流 -> 投影
    uPrev.set(u); vPrev.set(v);
    this.diffuse(BND_VERTICAL, u, uPrev, p.visc, dt);
    this.diffuse(BND_HORIZONTAL, v, vPrev, p.visc, dt);
    this.project();
    uPrev.set(u); vPrev.set(v);
    this.advect(BND_VERTICAL, u, uPrev, uPrev, vPrev, dt);
    this.advect(BND_HORIZONTAL, v, vPrev, uPrev, vPrev, dt);
    this.project();

    // 密度: 扩散 -> 平流 -> 衰减
    densPrev.set(dens);
    this.diffuse(BND_NONE, dens, densPrev, p.diff, dt);
    densPrev.set(dens);
    this.advect(BND_NONE, dens, densPrev, u, v, dt);
    if (p.fade > 0) {
      const k = Math.max(0, 1 - p.fade * dt);
      for (let i = 0; i < this.size; i++) dens[i] *= k;
    }
  }

  /** 诊断: 最大速度 / 是否含 NaN / 平均散度, 供测试与监控 */
  diagnostics() {
    let maxU = 0, nan = false, divSum = 0, mass = 0;
    const N = this.N;
    for (let j = 1; j <= N; j++) {
      for (let i = 1; i <= N; i++) {
        const idx = this.IX(i, j);
        const uu = this.u[idx], vv = this.v[idx];
        if (!Number.isFinite(uu) || !Number.isFinite(vv) || !Number.isFinite(this.dens[idx])) nan = true;
        const m = Math.abs(uu) + Math.abs(vv);
        if (m > maxU) maxU = m;
        divSum += Math.abs(uu - this.u[this.IX(i - 1, j)] + vv - this.v[this.IX(i, j - 1)]);
        mass += this.dens[idx];
      }
    }
    return { maxSpeed: maxU, hasNaN: nan, meanDivergence: divSum / (N * N), totalMass: mass };
  }
}
