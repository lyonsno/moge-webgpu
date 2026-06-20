// Multi-head self-attention compute shaders for DINOv2 ViT.
// Adapted from webgpu-samples visionTransformer with 2D dispatch.
//
// Three entry points:
//   computeScores: Q·K^T scaled dot product → scores
//   softmax: row-wise numerically stable softmax
//   applyAttn: scores @ V → output

struct ScoreParams {
  N: u32,        // number of tokens
  D: u32,        // model dimension
  numHeads: u32,
  headDim: u32,
  scale: f32,
  numWorkgroupsX: u32,
}

struct SoftmaxParams {
  N: u32,
  numHeads: u32,
  numWorkgroupsX: u32,
}

struct ApplyParams {
  N: u32,
  D: u32,
  numHeads: u32,
  headDim: u32,
  numWorkgroupsX: u32,
}

// --- Attention scores ---
@group(0) @binding(0) var<uniform> scoreParams: ScoreParams;
@group(0) @binding(1) var<storage, read> qBuf: array<f32>;
@group(0) @binding(2) var<storage, read> kBuf: array<f32>;
@group(0) @binding(3) var<storage, read_write> scoreBuf: array<f32>;

@compute @workgroup_size(256)
fn computeScores(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * scoreParams.numWorkgroupsX;
  let idx = linearWG * 256u + lid.x;

  let N = scoreParams.N;
  let numHeads = scoreParams.numHeads;
  let headDim = scoreParams.headDim;
  let D = scoreParams.D;
  let totalScores = numHeads * N * N;

  if (idx >= totalScores) { return; }

  let head = idx / (N * N);
  let remainder = idx % (N * N);
  let qi = remainder / N;
  let ki = remainder % N;
  let headOffset = head * headDim;

  var dot = 0.0;
  for (var d = 0u; d < headDim; d++) {
    dot += qBuf[qi * D + headOffset + d] * kBuf[ki * D + headOffset + d];
  }

  scoreBuf[idx] = dot * scoreParams.scale;
}

// --- Softmax ---
// Uses separate bind group with SoftmaxParams
@group(0) @binding(0) var<uniform> softmaxParams: SoftmaxParams;
@group(0) @binding(1) var<storage, read_write> softmaxScoreBuf: array<f32>;

@compute @workgroup_size(256)
fn softmax(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * softmaxParams.numWorkgroupsX;
  let idx = linearWG * 256u + lid.x;

  let N = softmaxParams.N;
  let totalRows = softmaxParams.numHeads * N;

  if (idx >= totalRows) { return; }

  let base = idx * N;

  // Find max
  var m = -1e30;
  for (var i = 0u; i < N; i++) {
    m = max(m, softmaxScoreBuf[base + i]);
  }

  // Exp and sum
  var s = 0.0;
  for (var i = 0u; i < N; i++) {
    let e = exp(softmaxScoreBuf[base + i] - m);
    softmaxScoreBuf[base + i] = e;
    s += e;
  }

  // Normalize
  for (var i = 0u; i < N; i++) {
    softmaxScoreBuf[base + i] = softmaxScoreBuf[base + i] / s;
  }
}

// --- Apply attention ---
@group(0) @binding(0) var<uniform> applyParams: ApplyParams;
@group(0) @binding(1) var<storage, read> applyScoreBuf: array<f32>;
@group(0) @binding(2) var<storage, read> vBuf: array<f32>;
@group(0) @binding(3) var<storage, read_write> attnOutput: array<f32>;

@compute @workgroup_size(256)
fn applyAttn(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * applyParams.numWorkgroupsX;
  let idx = linearWG * 256u + lid.x;

  let N = applyParams.N;
  let D = applyParams.D;
  let numHeads = applyParams.numHeads;
  let headDim = applyParams.headDim;

  if (idx >= N * D) { return; }

  let row = idx / D;
  let col = idx % D;
  let head = col / headDim;
  let d = col % headDim;

  var val = 0.0;
  let scoreBase = head * N * N + row * N;
  for (var j = 0u; j < N; j++) {
    val += applyScoreBuf[scoreBase + j] * vBuf[j * D + head * headDim + d];
  }
  attnOutput[idx] = val;
}
