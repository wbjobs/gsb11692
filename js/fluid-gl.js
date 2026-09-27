// WebGL2 Stable-Fluids 求解器
// 半拉格朗日平流(无条件稳定) + Jacobi 压力投影 + 显式边界条件 pass
// 速度场: 速度(ux2) / 染料(ux2) / 压力(ux2) / 散度(x1), 全部 RGBA16F ping-pong

const VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG_HEAD = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 vUv;
out vec4 outColor;
`;

const FRAG_ADVECT = FRAG_HEAD + `
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform float dt;
uniform float dissipation;
void main() {
  // 半拉格朗日回溯, 双线性采样由硬件 LINEAR 完成; 无条件稳定
  vec2 coord = vUv - dt * texture(uVelocity, vUv).xy;
  outColor = texture(uSource, coord) / (1.0 + dissipation * dt);
}`;

const FRAG_SPLAT = FRAG_HEAD + `
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec2 point;
uniform float radius;
void main() {
  vec2 p = vUv - point;
  p.x *= aspectRatio;
  vec3 splat = exp(-dot(p, p) / radius) * color;
  vec3 base = texture(uTarget, vUv).xyz;
  outColor = vec4(base + splat, 1.0);
}`;

const FRAG_DIVERGENCE = FRAG_HEAD + `
uniform sampler2D uVelocity;
uniform vec2 texelSize;
void main() {
  float L = texture(uVelocity, vUv - vec2(texelSize.x, 0.0)).x;
  float R = texture(uVelocity, vUv + vec2(texelSize.x, 0.0)).x;
  float B = texture(uVelocity, vUv - vec2(0.0, texelSize.y)).y;
  float T = texture(uVelocity, vUv + vec2(0.0, texelSize.y)).y;
  vec2 C = texture(uVelocity, vUv).xy;
  // 边界: 无滑移, 法向速度取反
  if (vUv.x - texelSize.x < 0.0) L = -C.x;
  if (vUv.x + texelSize.x > 1.0) R = -C.x;
  if (vUv.y - texelSize.y < 0.0) B = -C.y;
  if (vUv.y + texelSize.y > 1.0) T = -C.y;
  outColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
}`;

const FRAG_CLEAR = FRAG_HEAD + `
uniform sampler2D uTexture;
uniform float value;
void main() {
  outColor = value * texture(uTexture, vUv);
}`;

const FRAG_PRESSURE = FRAG_HEAD + `
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
uniform vec2 texelSize;
void main() {
  // CLAMP_TO_EDGE => 边界外采样等于边界值, 即 Neumann 边界 dp/dn = 0
  float L = texture(uPressure, vUv - vec2(texelSize.x, 0.0)).x;
  float R = texture(uPressure, vUv + vec2(texelSize.x, 0.0)).x;
  float B = texture(uPressure, vUv - vec2(0.0, texelSize.y)).x;
  float T = texture(uPressure, vUv + vec2(0.0, texelSize.y)).x;
  float divergence = texture(uDivergence, vUv).x;
  float p = (L + R + B + T - divergence) * 0.25;
  outColor = vec4(p, 0.0, 0.0, 1.0);
}`;

const FRAG_GRADIENT_SUBTRACT = FRAG_HEAD + `
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
uniform vec2 texelSize;
void main() {
  float L = texture(uPressure, vUv - vec2(texelSize.x, 0.0)).x;
  float R = texture(uPressure, vUv + vec2(texelSize.x, 0.0)).x;
  float B = texture(uPressure, vUv - vec2(0.0, texelSize.y)).x;
  float T = texture(uPressure, vUv + vec2(0.0, texelSize.y)).x;
  vec2 velocity = texture(uVelocity, vUv).xy;
  velocity -= 0.5 * vec2(R - L, T - B);
  outColor = vec4(velocity, 0.0, 1.0);
}`;

// 边界 pass: 内部直接拷贝, 边界从相邻内部格取值并乘以 scale
// scale = -1 => 速度无滑移; scale = 1 => Neumann
const FRAG_BOUNDARY = FRAG_HEAD + `
uniform sampler2D uTex;
uniform vec2 texelSize;
uniform float scale;
void main() {
  vec2 fc = gl_FragCoord.xy;
  vec2 size = 1.0 / texelSize;
  bool border = fc.x < 1.5 || fc.y < 1.5 || fc.x > size.x - 1.5 || fc.y > size.y - 1.5;
  if (!border) { outColor = texture(uTex, vUv); return; }
  vec2 offset = vec2(0.0);
  if (fc.x < 1.5) offset = vec2(1.0, 0.0);
  else if (fc.x > size.x - 1.5) offset = vec2(-1.0, 0.0);
  else if (fc.y < 1.5) offset = vec2(0.0, 1.0);
  else offset = vec2(0.0, -1.0);
  outColor = scale * texture(uTex, (fc + offset) * texelSize);
}`;

const FRAG_DISPLAY = FRAG_HEAD + `
uniform sampler2D uTexture;
void main() {
  vec3 c = texture(uTexture, vUv).rgb;
  // 简单色调映射, 防止过曝
  c = c / (1.0 + 0.15 * max(max(c.r, c.g), c.b));
  outColor = vec4(c, 1.0);
}`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error('Shader compile error: ' + gl.getShaderInfoLog(shader));
  }
  return shader;
}

class Program {
  constructor(gl, fragSource, sharedVert) {
    this.program = gl.createProgram();
    gl.attachShader(this.program, sharedVert);
    gl.attachShader(this.program, compileShader(gl, gl.FRAGMENT_SHADER, fragSource));
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error('Program link error: ' + gl.getProgramInfoLog(this.program));
    }
    this.uniforms = {};
    const count = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const name = gl.getActiveUniform(this.program, i).name;
      this.uniforms[name] = gl.getUniformLocation(this.program, name);
    }
    this.gl = gl;
  }
  bind() { this.gl.useProgram(this.program); }
}

function createFBO(gl, w, h, filter) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.viewport(0, 0, w, h);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  return {
    tex, fbo, w, h,
    attach(unit) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      return unit;
    },
    dispose() { gl.deleteTexture(tex); gl.deleteFramebuffer(fbo); },
  };
}

function createDoubleFBO(gl, w, h, filter) {
  return {
    read: createFBO(gl, w, h, filter),
    write: createFBO(gl, w, h, filter),
    swap() { const t = this.read; this.read = this.write; this.write = t; },
    dispose() { this.read.dispose(); this.write.dispose(); },
  };
}

export class FluidGL {
  constructor(canvas, simRes) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false, depth: false, stencil: false, antialias: false,
      preserveDrawingBuffer: false, powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 不可用');
    if (!gl.getExtension('EXT_color_buffer_float') &&
        !gl.getExtension('EXT_color_buffer_half_float')) {
      throw new Error('缺少浮点渲染目标扩展 (EXT_color_buffer_float)');
    }
    this.gl = gl;

    const vert = compileShader(gl, gl.VERTEX_SHADER, VERT);
    this.progAdvect   = new Program(gl, FRAG_ADVECT, vert);
    this.progSplat    = new Program(gl, FRAG_SPLAT, vert);
    this.progDiv      = new Program(gl, FRAG_DIVERGENCE, vert);
    this.progClear    = new Program(gl, FRAG_CLEAR, vert);
    this.progPressure = new Program(gl, FRAG_PRESSURE, vert);
    this.progGradSub  = new Program(gl, FRAG_GRADIENT_SUBTRACT, vert);
    this.progBoundary = new Program(gl, FRAG_BOUNDARY, vert);
    this.progDisplay  = new Program(gl, FRAG_DISPLAY, vert);

    // 全屏四边形
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    gl.disable(gl.BLEND);
    this.fields = null;
    this.setResolution(simRes);
  }

  setResolution(n) {
    if (this.fields) {
      for (const f of Object.values(this.fields)) f.dispose();
    }
    const gl = this.gl;
    this.simRes = n;
    this.texelSize = [1 / n, 1 / n];
    this.fields = {
      velocity:   createDoubleFBO(gl, n, n, gl.LINEAR),
      dye:        createDoubleFBO(gl, n, n, gl.LINEAR),
      pressure:   createDoubleFBO(gl, n, n, gl.NEAREST),
      divergence: createFBO(gl, n, n, gl.NEAREST),
    };
  }

  // 估算显存占用 (字节): 7 张 RGBA16F 纹理
  memoryBytes() { return 7 * this.simRes * this.simRes * 8; }

  blit(target) {
    const gl = this.gl;
    if (target == null) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      gl.viewport(0, 0, target.w, target.h);
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  boundary(field, scale) {
    const gl = this.gl, p = this.progBoundary;
    p.bind();
    gl.uniform1i(p.uniforms.uTex, field.read.attach(0));
    gl.uniform2f(p.uniforms.texelSize, this.texelSize[0], this.texelSize[1]);
    gl.uniform1f(p.uniforms.scale, scale);
    this.blit(field.write);
    field.swap();
  }

  splat(x, y, dx, dy, color, radius, aspectRatio) {
    const gl = this.gl, p = this.progSplat, F = this.fields;
    p.bind();
    gl.uniform1f(p.uniforms.aspectRatio, aspectRatio);
    gl.uniform2f(p.uniforms.point, x, y);
    gl.uniform1f(p.uniforms.radius, radius);
    // 速度
    gl.uniform1i(p.uniforms.uTarget, F.velocity.read.attach(0));
    gl.uniform3f(p.uniforms.color, dx, dy, 0);
    this.blit(F.velocity.write); F.velocity.swap();
    // 染料
    gl.uniform1i(p.uniforms.uTarget, F.dye.read.attach(0));
    gl.uniform3f(p.uniforms.color, color[0], color[1], color[2]);
    this.blit(F.dye.write); F.dye.swap();
  }

  step(dt, params) {
    const gl = this.gl, F = this.fields, ts = this.texelSize;

    // 1. 速度自平流
    let p = this.progAdvect;
    p.bind();
    gl.uniform1f(p.uniforms.dt, dt);
    gl.uniform1i(p.uniforms.uVelocity, F.velocity.read.attach(0));
    gl.uniform1i(p.uniforms.uSource, 0);
    gl.uniform1f(p.uniforms.dissipation, params.velDiss);
    this.blit(F.velocity.write); F.velocity.swap();
    this.boundary(F.velocity, -1); // 无滑移

    // 2. 散度
    p = this.progDiv;
    p.bind();
    gl.uniform1i(p.uniforms.uVelocity, F.velocity.read.attach(0));
    gl.uniform2f(p.uniforms.texelSize, ts[0], ts[1]);
    this.blit(F.divergence);

    // 3. 压力衰减(防止误差累积) + Jacobi 迭代
    p = this.progClear;
    p.bind();
    gl.uniform1i(p.uniforms.uTexture, F.pressure.read.attach(0));
    gl.uniform1f(p.uniforms.value, 0.8);
    this.blit(F.pressure.write); F.pressure.swap();

    p = this.progPressure;
    p.bind();
    gl.uniform1i(p.uniforms.uDivergence, F.divergence.attach(0));
    gl.uniform2f(p.uniforms.texelSize, ts[0], ts[1]);
    for (let i = 0; i < params.iters; i++) {
      gl.uniform1i(p.uniforms.uPressure, F.pressure.read.attach(1));
      this.blit(F.pressure.write); F.pressure.swap();
    }

    // 4. 减去压力梯度 -> 无散速度场
    p = this.progGradSub;
    p.bind();
    gl.uniform2f(p.uniforms.texelSize, ts[0], ts[1]);
    gl.uniform1i(p.uniforms.uPressure, F.pressure.read.attach(0));
    gl.uniform1i(p.uniforms.uVelocity, F.velocity.read.attach(1));
    this.blit(F.velocity.write); F.velocity.swap();
    this.boundary(F.velocity, -1);

    // 5. 染料随速度平流
    p = this.progAdvect;
    p.bind();
    gl.uniform1i(p.uniforms.uVelocity, F.velocity.read.attach(0));
    gl.uniform1i(p.uniforms.uSource, F.dye.read.attach(1));
    gl.uniform1f(p.uniforms.dissipation, params.dyeDiss);
    this.blit(F.dye.write); F.dye.swap();
  }

  render() {
    const gl = this.gl, p = this.progDisplay;
    p.bind();
    gl.uniform1i(p.uniforms.uTexture, this.fields.dye.read.attach(0));
    this.blit(null);
  }

  clear() { this.setResolution(this.simRes); }
}
