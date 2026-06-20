// Linear projection: output = input @ weight + bias
// Adapted from webgpu-samples visionTransformer mlp.wgsl with 2D dispatch.
// Weight layout: [inDim, outDim] (row-major, transposed from PyTorch convention)

struct Params {
  numRows: u32,
  inDim: u32,
  outDim: u32,
  numWorkgroupsX: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 256;

@compute @workgroup_size(WG_SIZE)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * WG_SIZE + lid.x;

  if (idx >= params.numRows * params.outDim) { return; }

  let row = idx / params.outDim;
  let col = idx % params.outDim;

  var val = bias[col];
  for (var k = 0u; k < params.inDim; k++) {
    val += input[row * params.inDim + k] * weight[k * params.outDim + col];
  }
  output[idx] = val;
}
