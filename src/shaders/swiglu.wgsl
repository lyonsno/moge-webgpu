// SwiGLU FFN compute shader for DINOv2.
//
// DINOv2 uses SwiGLU instead of GELU MLP:
//   x12 = w12(x)         — linear: [D] → [2*hiddenDim]
//   x1, x2 = split(x12)  — split in half
//   hidden = silu(x1) * x2  — gated activation
//   output = w3(hidden)   — linear: [hiddenDim] → [D]
//
// Weight layout:
//   w12_weight: [D, 2*hiddenDim] (transposed for row-major matmul)
//   w12_bias:   [2*hiddenDim]
//   w3_weight:  [hiddenDim, D]
//   w3_bias:    [D]

struct Params {
  numRows: u32,    // number of tokens
  D: u32,          // model dim (1024)
  hiddenDim: u32,  // hidden dim (2730 for ViT-L DINOv2)
  numWorkgroupsX: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> w12_weight: array<f32>;
@group(0) @binding(3) var<storage, read> w12_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> hidden: array<f32>;

const WG_SIZE: u32 = 256;

fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

// Pass 1: w12 projection + SwiGLU gating → hidden
@compute @workgroup_size(WG_SIZE)
fn swiglu_gate(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * WG_SIZE + lid.x;
  let totalWork = params.numRows * params.hiddenDim;

  if (idx >= totalWork) { return; }

  let row = idx / params.hiddenDim;
  let col = idx % params.hiddenDim;
  let doubleHidden = params.hiddenDim * 2u;

  // Compute w12 for both halves at this column
  var val1 = w12_bias[col];
  var val2 = w12_bias[params.hiddenDim + col];

  for (var k = 0u; k < params.D; k++) {
    let inp = input[row * params.D + k];
    val1 += inp * w12_weight[k * doubleHidden + col];
    val2 += inp * w12_weight[k * doubleHidden + params.hiddenDim + col];
  }

  // SwiGLU: silu(x1) * x2
  hidden[idx] = silu(val1) * val2;
}

// Pass 2 uses the mlp.wgsl linear shader (w3 projection)
