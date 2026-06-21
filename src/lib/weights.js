/**
 * weights.js — Load MoGe-2 weights from flat binary format.
 *
 * Binary format (from convert_weights.py):
 *   Header: 4 (magic) + 4 (version) + 4 (num_tensors) + 4 (header_size) = 16 bytes
 *   Tensor table: num_tensors × 96 bytes each
 *     64 bytes: name (null-padded ASCII)
 *     4 bytes: dtype (0=fp32, 1=fp16)
 *     4 bytes: ndim
 *     16 bytes: shape (4 x u32)
 *     4 bytes: offset
 *     4 bytes: size
 *   Weight data: packed tensors
 *
 * Maps PyTorch state_dict names to the dispatch chain's weight structure.
 */

import { createStorageBuffer } from './gpu.js';

const MAGIC = 0x45474F4D; // "MOGE" in little-endian
const ENTRY_SIZE = 96;

/**
 * Parse the binary header and tensor table.
 * Returns { tensors: Map<name, { dtype, shape, offset, size }> }
 */
function parseHeader(buffer) {
  const view = new DataView(buffer);

  const magic = view.getUint32(0, true);
  if (magic !== MAGIC) {
    throw new Error(`Invalid weight file magic: 0x${magic.toString(16)}`);
  }

  const version = view.getUint32(4, true);
  if (version !== 1) {
    throw new Error(`Unsupported weight file version: ${version}`);
  }

  const numTensors = view.getUint32(8, true);
  const headerSize = view.getUint32(12, true);

  const tensors = new Map();
  for (let i = 0; i < numTensors; i++) {
    const entryOffset = 16 + i * ENTRY_SIZE;

    // Name (64 bytes, null-terminated ASCII)
    const nameBytes = new Uint8Array(buffer, entryOffset, 64);
    let nameEnd = nameBytes.indexOf(0);
    if (nameEnd === -1) nameEnd = 64;
    const name = new TextDecoder().decode(nameBytes.slice(0, nameEnd));

    const dtype = view.getUint32(entryOffset + 64, true);
    const ndim = view.getUint32(entryOffset + 68, true);
    const shape = [];
    for (let d = 0; d < ndim; d++) {
      shape.push(view.getUint32(entryOffset + 72 + d * 4, true));
    }
    const offset = view.getUint32(entryOffset + 88, true);
    const size = view.getUint32(entryOffset + 92, true);

    tensors.set(name, { dtype, shape, offset, size });
  }

  return { tensors, headerSize };
}

/**
 * Extract a tensor from the binary buffer as a GPU storage buffer.
 * Converts fp16 → fp32 on CPU before uploading (WebGPU storage buffers are fp32).
 */
function extractTensor(device, buffer, tensorInfo) {
  const { dtype, offset, size } = tensorInfo;

  if (dtype === 0) {
    // fp32
    const data = new Float32Array(buffer, offset, size / 4);
    return createStorageBuffer(device, data);
  } else {
    // fp16 → fp32
    const fp16 = new Uint16Array(buffer, offset, size / 2);
    const fp32 = new Float32Array(fp16.length);
    for (let i = 0; i < fp16.length; i++) {
      fp32[i] = fp16ToFp32(fp16[i]);
    }
    return createStorageBuffer(device, fp32);
  }
}

/**
 * Convert fp16 (as uint16) to fp32.
 */
function fp16ToFp32(h) {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;

  if (exp === 0) {
    if (mant === 0) return sign ? -0.0 : 0.0;
    // Subnormal
    let val = mant / 1024.0 * Math.pow(2, -14);
    return sign ? -val : val;
  }
  if (exp === 31) {
    return mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }

  const val = Math.pow(2, exp - 15) * (1 + mant / 1024.0);
  return sign ? -val : val;
}

/**
 * Extract tensor data as CPU Float32Array (no GPU upload).
 */
function extractTensorCPU(buffer, tensorInfo) {
  const { dtype, offset, size } = tensorInfo;
  if (dtype === 0) {
    return new Float32Array(buffer.slice(offset, offset + size));
  } else {
    const fp16 = new Uint16Array(buffer, offset, size / 2);
    const fp32 = new Float32Array(fp16.length);
    for (let i = 0; i < fp16.length; i++) {
      fp32[i] = fp16ToFp32(fp16[i]);
    }
    return fp32;
  }
}

/**
 * Get a tensor by name, or return null if not found.
 */
function getTensor(device, buffer, tensors, name) {
  const info = tensors.get(name);
  if (!info) return null;
  return extractTensor(device, buffer, info);
}

/**
 * Build the full weight structure from the binary file.
 * Maps PyTorch state_dict names to the dispatch chain's expected format.
 */
export async function loadWeights(device, url, onProgress) {
  // Fetch the binary file
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch weights: ${response.status}`);
  }

  const contentLength = parseInt(response.headers.get('content-length') || '0');
  const reader = response.body.getReader();

  // Read with progress
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress) {
      onProgress(received, contentLength);
    }
  }

  // Concatenate chunks
  const buffer = new ArrayBuffer(received);
  const uint8 = new Uint8Array(buffer);
  let pos = 0;
  for (const chunk of chunks) {
    uint8.set(chunk, pos);
    pos += chunk.length;
  }

  const { tensors } = parseHeader(buffer);

  // Helper: get tensor or throw
  const get = (name) => {
    const buf = getTensor(device, buffer, tensors, name);
    if (!buf) throw new Error(`Missing weight: ${name}`);
    return buf;
  };

  // Helper: get tensor or return zero buffer
  const getOrZeros = (name, size) => {
    const buf = getTensor(device, buffer, tensors, name);
    if (buf) return buf;
    return createStorageBuffer(device, new Float32Array(size));
  };

  // Build ConvStack weights for a given prefix (neck, points_head, etc.)
  function buildConvStackWeights(prefix, config) {
    const { dimIn, dimResBlocks, dimOut, numResBlocks, resamplers } = config;

    return {
      levels: dimResBlocks.map((dimRB, i) => {
        // input_block: 1x1 conv
        const inputWeight = dimIn[i] != null
          ? get(`${prefix}.input_blocks.${i}.weight`)
          : null;
        const inputBias = dimIn[i] != null
          ? get(`${prefix}.input_blocks.${i}.bias`)
          : null;

        // res_blocks
        const resBlocks = [];
        for (let j = 0; j < numResBlocks[i]; j++) {
          // ResidualConvBlock layers: [norm0, act1, conv2, norm3, act4, conv5]
          // With norm='none', layers.0 and layers.3 are Identity (no weights)
          resBlocks.push({
            norm1_scale: null, // norm='none'
            norm1_bias: null,
            conv1_weight: get(`${prefix}.res_blocks.${i}.${j}.layers.2.weight`),
            conv1_bias: get(`${prefix}.res_blocks.${i}.${j}.layers.2.bias`),
            norm2_scale: null,
            norm2_bias: null,
            conv2_weight: get(`${prefix}.res_blocks.${i}.${j}.layers.5.weight`),
            conv2_bias: get(`${prefix}.res_blocks.${i}.${j}.layers.5.bias`),
            skip_weight: null, // in_channels == out_channels for all MoGe-2 res blocks
          });
        }

        // output_block: 1x1 conv (only at final level with dimOut != null)
        const outputWeight = dimOut[i] != null
          ? get(`${prefix}.output_blocks.${i}.weight`)
          : null;
        const outputBias = dimOut[i] != null
          ? get(`${prefix}.output_blocks.${i}.bias`)
          : null;

        // resampler
        let resampler = null;
        if (i < dimResBlocks.length - 1 && resamplers[i]) {
          if (resamplers[i] === 'conv_transpose') {
            resampler = {
              deconv_weight: get(`${prefix}.resamplers.${i}.0.weight`),
              deconv_bias: get(`${prefix}.resamplers.${i}.0.bias`),
              conv_weight: get(`${prefix}.resamplers.${i}.1.weight`),
              conv_bias: get(`${prefix}.resamplers.${i}.1.bias`),
            };
          } else if (resamplers[i] === 'bilinear') {
            // Bilinear resampler: no .0 (upsample is parameterfree), only .1 (conv)
            resampler = {
              conv_weight: get(`${prefix}.resamplers.${i}.1.weight`),
              conv_bias: get(`${prefix}.resamplers.${i}.1.bias`),
            };
          }
        }

        return {
          input_weight: inputWeight,
          input_bias: inputBias,
          res_blocks: resBlocks,
          output_weight: outputWeight,
          output_bias: outputBias,
          resampler,
        };
      }),
    };
  }

  // Build encoder weights (for when ViT backbone is implemented)
  // For now, just extract the output projections and image normalization
  const encoder = {
    imageMean: get('encoder.image_mean'),
    imageStd: get('encoder.image_std'),
    outputProjections: [0, 1, 2, 3].map(i => ({
      weight: get(`encoder.output_projections.${i}.weight`),
      bias: get(`encoder.output_projections.${i}.bias`),
    })),
    patchEmbed: {
      weight: get('encoder.backbone.patch_embed.proj.weight'),
      bias: get('encoder.backbone.patch_embed.proj.bias'),
    },
    posEmbed: get('encoder.backbone.pos_embed'),
    clsToken: get('encoder.backbone.cls_token'),
    norm: {
      weight: get('encoder.backbone.norm.weight'),
      bias: get('encoder.backbone.norm.bias'),
    },
    blockWeights: {},
  };

  // Load all 24 transformer block weights
  for (let l = 0; l < 24; l++) {
    const prefix = `encoder.backbone.blocks.${l}`;
    for (const name of [
      'attn.qkv.weight', 'attn.qkv.bias',
      'attn.proj.weight', 'attn.proj.bias',
      'norm1.weight', 'norm1.bias',
      'norm2.weight', 'norm2.bias',
      'ls1.gamma', 'ls2.gamma',
      'mlp.fc1.weight', 'mlp.fc1.bias',
      'mlp.fc2.weight', 'mlp.fc2.bias',
    ]) {
      const fullName = `${prefix}.${name}`;
      const buf = getTensor(device, buffer, tensors, fullName);
      if (buf) {
        encoder.blockWeights[fullName] = buf;
      }
    }
  }

  // Build all ConvStack weights
  const MODEL_CONFIG = (await import('./inference.js')).default;

  // Import config directly
  const neckConfig = {
    dimIn: [1026, 2, 2, 2, 2],
    dimResBlocks: [1024, 256, 128, 64, 32],
    dimOut: [null, null, null, null, null],
    numResBlocks: [0, 2, 2, 2, 0],
    resamplers: ['conv_transpose', 'conv_transpose', 'conv_transpose', 'bilinear'],
  };

  const headConfig = {
    dimIn: [1024, 256, 128, 64, 32],
    dimResBlocks: [1024, 256, 128, 64, 32],
    numResBlocks: [0, 1, 1, 1, 0],
    resamplers: ['conv_transpose', 'conv_transpose', 'conv_transpose', 'bilinear'],
  };

  const weights = {
    encoder,
    neck: buildConvStackWeights('neck', neckConfig),
    pointsHead: buildConvStackWeights('points_head', {
      ...headConfig,
      dimOut: [null, null, null, null, 3],
    }),
    // NOTE: moge-2-vitl has no normal_head — normals derived from point map
    maskHead: buildConvStackWeights('mask_head', {
      ...headConfig,
      dimOut: [null, null, null, null, 1],
    }),
    scaleHead: {
      layers: [
        { weight: extractTensorCPU(buffer, tensors.get('scale_head.0.weight')), bias: extractTensorCPU(buffer, tensors.get('scale_head.0.bias')), inDim: 1024, outDim: 1024 },
        { weight: extractTensorCPU(buffer, tensors.get('scale_head.2.weight')), bias: extractTensorCPU(buffer, tensors.get('scale_head.2.bias')), inDim: 1024, outDim: 1024 },
        { weight: extractTensorCPU(buffer, tensors.get('scale_head.4.weight')), bias: extractTensorCPU(buffer, tensors.get('scale_head.4.bias')), inDim: 1024, outDim: 1 },
      ],
    },
  };

  console.log(`Loaded ${tensors.size} tensors from weight file`);
  return weights;
}
