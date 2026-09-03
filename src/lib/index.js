/**
 * Library entry for embedding MoGe-2 WebGPU inference in a host application.
 *
 * The host owns the GPUDevice (share it with a renderer/simulation) and passes
 * it to MoGeInference; weights stream from a local /weights.bin when present,
 * otherwise from the hosted HuggingFace copy. Cooperative scheduling is
 * requested per run via options.scheduler ({ mode: 'cooperative', yieldMs,
 * vitBlockChunkSize, waitForSubmittedWorkDone }).
 */
export { MoGeInference } from './inference.js';
export { initGPU, inferenceLimits, INFERENCE_LIMIT_KEYS } from './gpu.js';
export {
  resolveCooperativeScheduler,
  createMogeSchedulerVerificationReceipt,
} from './scheduler_receipt.js';
