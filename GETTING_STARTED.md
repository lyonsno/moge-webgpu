# MoGe In Your Application

MoGe turns a photograph into depth, surface normals, and a 3D point map on your application's GPU. Install the package and pass it your application's GPU device. Hosted weights load by default; you can supply your own weight URL. Reuse the model across images, then dispose of its resources without destroying your device.

## Install

```sh
npm install @kaminos/moge
```

```js
import { MoGeInference } from '@kaminos/moge';
```

## Load, Run, Release

`gpu` is your application's `{ device, adapter, backendIdentity }` context. `images` contains RGBA images shaped as `{ width, height, data }`. Each result includes `depth`, `normals`, `points`, `colors`, `width`, and `height`.

```js
import { MoGeInference } from '@kaminos/moge';

export async function inferDepth(gpu, images, { weightsUrl, onProgress } = {}) {
  const inference = new MoGeInference(gpu, { weightsUrl });
  try {
    await inference.init(onProgress);
    const outputs = [];
    for await (const image of images) {
      outputs.push(await inference.run(image, {
        scheduler: {
          mode: 'cooperative', splitVitBlocks: true,
          splitDecoderResBlocks: true, pacing: 'bounded-prefix',
        },
      }));
    }
    return outputs;
  } finally {
    await inference.dispose();
  }
}
```

For an application creating a new WebGPU device, the package exports the device helper:

```js
import { initGPU } from '@kaminos/moge';

const gpu = await initGPU();
```

`onProgress` receives downloaded and total weight bytes. `weightsUrl` accepts the URL of a MoGe weight file. Loading and initialization errors reject rather than substituting synthetic output. Serve the application over HTTPS or localhost.

For rendering alongside inference, use the kit's [shared-device getting-started example](https://github.com/lyonsno/kaminos/blob/main/webgpu-inference-kit/docs/getting-started.md) and the [Material Photograph composition](https://github.com/lyonsno/kaminos/blob/main/demos/material-photo/main.js).

## Build From Source

From the MoGe WebGPU checkout:

```sh
npm ci
npm run build:lib
```

Output: `dist-lib/moge-inference.js`. For a direct-module application, replace the package import with the URL where you serve that file. Include `LICENSE` and `THIRD_PARTY_NOTICES.txt` when distributing it.
