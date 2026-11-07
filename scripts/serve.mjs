// Local dev server: cross-origin isolation for wasm threads, files served straight from D:.
// `no-store` keeps the browser from also caching the 570 MB model in its profile on C:.
import { createReadStream, promises as fs } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const PORT = Number(process.env.PORT ?? 8377);
const MOUNTS = [
  ['/vendor/ort/', join(ROOT, 'node_modules', 'onnxruntime-web', 'dist')],
  ['/models/', join(ROOT, 'models')],
  ['/', join(ROOT, 'web')],
];
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
};

function resolvePath(urlPath) {
  for (const [prefix, dir] of MOUNTS) {
    if (urlPath.startsWith(prefix)) {
      const file = normalize(join(dir, urlPath.slice(prefix.length)));
      return file === dir || file.startsWith(dir + sep) ? file : null;
    }
  }
  return null;
}

async function saveResult(req, res, name) {
  if (!/^[\w.-]+\.json$/.test(name)) {
    res.writeHead(400).end('bad name');
    return;
  }
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  await fs.mkdir(join(ROOT, 'results'), { recursive: true });
  await fs.writeFile(join(ROOT, 'results', name), Buffer.concat(chunks));
  res.writeHead(204).end();
}

createServer(async (req, res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cache-Control', 'no-store');
  try {
    const urlPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (req.method === 'PUT' && urlPath.startsWith('/results/')) {
      await saveResult(req, res, urlPath.slice('/results/'.length));
      return;
    }
    let file = resolvePath(urlPath === '/' ? '/index.html' : urlPath);
    if (!file) {
      res.writeHead(403).end();
      return;
    }
    const stat = await fs.stat(file);
    if (!stat.isFile()) {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, {
      'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream',
      'Content-Length': stat.size,
    });
    // Stream errors arrive after the try block has returned; unhandled, they stop the server.
    createReadStream(file)
      .on('error', (error) => res.destroy(error))
      .pipe(res);
  } catch (error) {
    res.writeHead(error.code === 'ENOENT' ? 404 : 500).end(String(error.message));
  }
}).listen(PORT, '127.0.0.1', () => console.log(`http://localhost:${PORT}`));
