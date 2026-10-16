/**
 * Library entry for embedding MoGe-2 WebGPU inference in a host application.
 *
 * The host owns the GPUDevice (share it with a renderer/simulation) and passes
 * it to MoGeInference. The optional second argument selects weightsUrl;
 * otherwise weights stream from the hosted HuggingFace copy. Loading errors
 * reject init(); diagnosticStub must be selected explicitly. Scheduling is
 * requested per run via options.scheduler ({ mode: 'cooperative', yieldMs,
 * vitBlockChunkSize, waitForSubmittedWorkDone }).
 */
export { MoGeInference } from './inference.js';
export {
  initGPU,
  inferenceLimits,
  INFERENCE_LIMIT_KEYS,
  buildMogeDeviceRequest,
  borrowedDeviceBackendIdentity,
} from './gpu.js';
export {
  resolveCooperativeScheduler,
  createMogeSchedulerVerificationReceipt,
} from './scheduler_receipt.js';
