import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { parseArgs, promisify } from 'node:util';
import { pathToFileURL } from 'node:url';
import { build } from 'vite';

const exec = promisify(execFile);

export async function packMoGeModule({ out, version = '0.1.0' } = {}) {
  if (typeof out !== 'string' || !out.trim()) throw new Error('MoGe package output directory is required');
  const root = path.resolve(import.meta.dirname, '..');
  const destination = path.resolve(out);
  await fs.mkdir(destination, { recursive: true });
  const stage = await fs.mkdtemp(path.join(os.tmpdir(), 'moge-pack-'));
  try {
    await build({ root, configFile: path.join(root, 'vite.lib.config.js'), build: { outDir: stage }, logLevel: 'error' });
    for (const file of ['README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.txt']) await fs.copyFile(path.join(root, file), path.join(stage, file));
    const manifest = {
      name: '@kaminos/moge', version, private: true, type: 'module', license: 'MIT',
      description: 'MoGe-2 depth estimation running entirely in WebGPU compute shaders',
      main: './moge-inference.js',
      exports: { '.': './moge-inference.js', './package.json': './package.json' },
      files: ['moge-inference.js', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.txt'],
      repository: { type: 'git', url: 'git+https://github.com/lyonsno/moge-webgpu.git' },
      keywords: ['webgpu', 'moge', 'depth-estimation', 'surface-normals', '3d-reconstruction'],
    };
    await fs.writeFile(path.join(stage, 'package.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    const { stdout } = await exec('npm', ['pack', '--json', '--ignore-scripts', '--pack-destination', destination], { cwd: stage });
    const [packed] = JSON.parse(stdout);
    return { name: packed.name, version: packed.version, tarball: path.join(destination, packed.filename),
      integrity: packed.integrity, shasum: packed.shasum, files: packed.files };
  } finally {
    await fs.rm(stage, { recursive: true, force: true });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const { values } = parseArgs({ options: { out: { type: 'string' }, version: { type: 'string' }, json: { type: 'boolean' } } });
  const packed = await packMoGeModule(values);
  console.log(values.json ? JSON.stringify(packed) : `packed ${packed.tarball}`);
}
