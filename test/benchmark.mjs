/**
 * 分辨率 vs 性能 基准: node test/benchmark.mjs
 * 输出各网格规模下的单步耗时 / 内存 / 理论可承载帧率, 用于量化取舍。
 */
import { FluidSolver } from '../js/solver.js';

const PARAMS = { visc: 0.0001, diff: 0.00001, vorticity: 2.0, fade: 0.15 };
const DT = 1 / 60;
const WARMUP = 20, MEASURE = 120;

console.log(' N    | 内存     | 平均步耗时 | 峰值步耗时 | 理论FPS上限');
console.log('------+----------+-----------+-----------+-----------');
for (const N of [64, 96, 128, 192, 256, 384]) {
  const s = new FluidSolver(N);
  s.splat(N / 2, N / 2, N / 16, 30, -30, 2);
  for (let i = 0; i < WARMUP; i++) s.step(DT, PARAMS);
  let total = 0, peak = 0;
  for (let i = 0; i < MEASURE; i++) {
    const t0 = performance.now();
    s.step(DT, PARAMS);
    const ms = performance.now() - t0;
    total += ms;
    if (ms > peak) peak = ms;
  }
  const avg = total / MEASURE;
  const mem = (s.memoryBytes() / 1048576).toFixed(2) + ' MB';
  const fpsCap = avg > 0 ? Math.floor(1000 / avg) : '∞';
  console.log(
    ` ${String(N).padEnd(4)} | ${mem.padEnd(8)} | ${avg.toFixed(2).padStart(6)} ms | ${peak.toFixed(2).padStart(6)} ms | ${String(fpsCap).padStart(5)}`
  );
}
console.log('\n注: 60 FPS 需要步耗时 < 16.7ms; 超出时主程序会自动降低分辨率。');
