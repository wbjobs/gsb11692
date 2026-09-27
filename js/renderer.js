/**
 * renderer.js — 渲染层。
 *
 * 优先 WebGL1: 密度场作为 LUMINANCE 纹理上传, fragment shader 做 inferno 风格
 * 伪彩色映射 + 简单光照浮雕, GPU 双线性放大。
 * 不支持 WebGL 时降级 Canvas 2D (putImageData + drawImage 缩放)。
 */

const VERT = `
attribute vec2 aPos;
varying vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG = `
precision mediump float;
varying vec2 vUV;
uniform sampler2D uDens;
uniform vec2 uTexel;

// inferno 近似调色板
vec3 inferno(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.00, 0.00, 0.02);
  vec3 c1 = vec3(0.25, 0.05, 0.43);
  vec3 c2 = vec3(0.76, 0.16, 0.22);
  vec3 c3 = vec3(0.98, 0.62, 0.04);
  vec3 c4 = vec3(0.99, 0.98, 0.80);
  vec3 c = mix(c0, c1, smoothstep(0.0, 0.25, t));
  c = mix(c, c2, smoothstep(0.25, 0.5, t));
  c = mix(c, c3, smoothstep(0.5, 0.8, t));
  c = mix(c, c4, smoothstep(0.8, 1.0, t));
  return c;
}

void main() {
  float d = texture2D(uDens, vUV).r;
  // 用密度梯度做法线, 加一点点方向光让烟雾有体积感
  float dx = texture2D(uDens, vUV + vec2(uTexel.x, 0.0)).r
           - texture2D(uDens, vUV - vec2(uTexel.x, 0.0)).r;
  float dy = texture2D(uDens, vUV + vec2(0.0, uTexel.y)).r
           - texture2D(uDens, vUV - vec2(0.0, uTexel.y)).r;
  vec3 n = normalize(vec3(-dx, dy, 0.35));
  float light = 0.75 + 0.25 * dot(n, normalize(vec3(-0.4, 0.6, 0.7)));
  vec3 col = inferno(d) * light;
  gl_FragColor = vec4(col, 1.0);
}`;

export class Renderer {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this.canvas = canvas;
    this.mode = '2d';
    this.gl = null;
    this.tex = null;
    this.prog = null;
    this.imageData = null;
    this.offscreen = null;
    this.ctx2d = null;
    this._initGL();
    if (!this.gl) this._init2D();
  }

  _initGL() {
    const gl = this.canvas.getContext('webgl', { antialias: false, alpha: false })
            || this.canvas.getContext('experimental-webgl');
    if (!gl) return;
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.warn('shader error:', gl.getShaderInfoLog(s));
        return null;
      }
      return s;
    };
    const vs = compile(gl.VERTEX_SHADER, VERT);
    const fs = compile(gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) return;
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'aPos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.uTexel = gl.getUniformLocation(prog, 'uTexel');
    this.gl = gl;
    this.prog = prog;
    this.mode = 'webgl';
  }

  _init2D() {
    this.ctx2d = this.canvas.getContext('2d');
    this.offscreen = document.createElement('canvas');
    this.offCtx = this.offscreen.getContext('2d');
    this.mode = '2d';
  }

  /**
   * 渲染一帧。
   * @param {Uint8Array} pixels N*N 灰度密度 (0..255)
   * @param {number} N 网格边长
   */
  draw(pixels, N) {
    if (this.mode === 'webgl') {
      const gl = this.gl;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, N, N, 0,
                    gl.LUMINANCE, gl.UNSIGNED_BYTE, pixels);
      gl.uniform2f(this.uTexel, 1 / N, 1 / N);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    } else {
      if (!this.imageData || this.imageData.width !== N) {
        this.offscreen.width = N;
        this.offscreen.height = N;
        this.imageData = this.offCtx.createImageData(N, N);
      }
      const out = this.imageData.data;
      for (let i = 0, o = 0; i < pixels.length; i++, o += 4) {
        const d = pixels[i];
        // 简化调色: 黑 -> 橙红 -> 白
        out[o]     = Math.min(255, d * 2.2);
        out[o + 1] = Math.min(255, d * 1.1);
        out[o + 2] = Math.min(255, d * 0.4 + (d > 200 ? (d - 200) * 3 : 0));
        out[o + 3] = 255;
      }
      this.offCtx.putImageData(this.imageData, 0, 0);
      this.ctx2d.imageSmoothingEnabled = true;
      this.ctx2d.drawImage(this.offscreen, 0, 0, this.canvas.width, this.canvas.height);
    }
  }
}
