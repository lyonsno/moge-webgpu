import { defineConfig } from 'vite';

// Library build: a single self-contained ESM bundle (WGSL inlined) that host
// applications import to run MoGe-2 inference on a device they own.
// Usage: npx vite build --config vite.lib.config.js
export default defineConfig({
  publicDir: false, // don't copy the 660MB weights into the lib output
  build: {
    target: 'esnext',
    outDir: 'dist-lib',
    lib: {
      entry: 'src/lib/index.js',
      formats: ['es'],
      fileName: 'moge-inference',
    },
    minify: false,
  },
});
