// conv_transpose2d_stride2.wgsl — specialized k=2, stride=2 transposed conv
//
// MoGe-2's ConvStack deconv resamplers use kernel_size=stride=2. In that case
// each output pixel maps to exactly one input pixel and one 2x2 kernel phase, so
// the general modulo/loop path can be collapsed to a single phase lookup.

struct ConvTransposeStride2Params {
  inC: u32,
  inH: u32,
  inW: u32,
  outC: u32,
  outH: u32,
  outW: u32,
  hasBias: u32,
  numWorkgroupsX: u32,
};

@group(0) @binding(0) var<uniform> params: ConvTransposeStride2Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn conv_transpose2d_stride2_main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let totalOut = params.outC * params.outH * params.outW;
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * WG_SIZE + lid.x;

  if (idx >= totalOut) {
    return;
  }

  let outSpatial = params.outH * params.outW;
  let oc = idx / outSpatial;
  let rem = idx % outSpatial;
  let oy = rem / params.outW;
  let ox = rem % params.outW;

  let iy = oy >> 1u;
  let ix = ox >> 1u;
  let ky = oy & 1u;
  let kx = ox & 1u;
  let kernelPhase = ky * 2u + kx;
  let inputSpatial = params.inH * params.inW;

  var sum: f32 = 0.0;
  for (var ic: u32 = 0u; ic < params.inC; ic++) {
    let inputIdx = ic * inputSpatial + iy * params.inW + ix;
    let weightIdx = ic * params.outC * 4u + oc * 4u + kernelPhase;
    sum += input[inputIdx] * weight[weightIdx];
  }

  if (params.hasBias != 0u) {
    sum += bias[oc];
  }

  output[idx] = sum;
}
