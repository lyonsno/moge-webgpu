import { defineConfig } from 'vite';

export default defineConfig({
  // GitHub Pages serves the site under /moge-webgpu/
  base: process.env.GITHUB_PAGES ? '/moge-webgpu/' : '/',
  server: {
    port: 5174,
    open: true,
  },
  build: {
    target: 'esnext',
  },
});
