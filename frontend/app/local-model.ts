export type LocalModelStatus = {
  downloaded: boolean;
  downloading: boolean;
  totalBytes: number;
  downloadedBytes: number;
  percent: number;
  speedBytesPerSecond: number;
  error: string;
};

export type LocalAdapterStatus = {
  webGpuAvailable: boolean;
  adapterName: string;
  intelSelected: boolean;
};

const MODEL_CACHE = "kalapaktts-vadom-v1";
const MODEL_MANIFEST_URL = "/model-manifest.json";
const MANIFEST_URL = "https://huggingface.co/Uddom/voxcpm2-khmer-webgpu/raw/main/manifest.json";

type ManifestFile = {
  name: string;
  path: string;
  bytes: number;
  sha256: string;
  externalData?: Array<{ path: string; bytes: number; sha256: string }>;
};

type Manifest = {
  id: string;
  version: string;
  baseUrl: string;
  files: ManifestFile[];
  tokenizer: string;
  tokenizerBytes: number;
  config: {
    totalBytes: number;
  };
};

export async function getLocalAdapterStatus(): Promise<LocalAdapterStatus> {
  const gpuNavigator = navigator as Navigator & {
    gpu?: {
      requestAdapter: (options?: { powerPreference?: "low-power" | "high-performance" }) => Promise<{
        info?: { device?: string; description?: string };
      } | null>;
    };
  };
  if (!gpuNavigator.gpu) {
    return { webGpuAvailable: false, adapterName: "", intelSelected: false };
  }

  try {
    const adapter = await gpuNavigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      return { webGpuAvailable: false, adapterName: "", intelSelected: false };
    }
    const adapterName = adapter.info?.device ?? adapter.info?.description ?? "WebGPU adapter detected; browser did not expose a name";
    const intelSelected = /intel|uhd|iris/i.test(adapterName);
    return { webGpuAvailable: true, adapterName, intelSelected };
  } catch {
    return { webGpuAvailable: false, adapterName: "", intelSelected: false };
  }
}

export async function getManifest(): Promise<Manifest> {
  const response = await fetch(MODEL_MANIFEST_URL, { cache: "no-cache" });
  if (!response.ok) throw new Error("The bundled local-model manifest is unavailable.");
  return response.json();
}

export async function getModelStorageInfo() {
  let usage = 0;
  let quota = 0;
  if (navigator.storage?.estimate) {
    const estimate = await navigator.storage.estimate();
    usage = estimate.usage ?? 0;
    quota = estimate.quota ?? 0;
  }
  return { usage, quota };
}

export async function requestPersistentStorage() {
  if (!navigator.storage?.persist) return false;
  return navigator.storage.persist();
}

function sha256Hex(buffer: ArrayBuffer) {
  return crypto.subtle.digest("SHA-256", buffer).then((digest) =>
    Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(""),
  );
}

export async function isModelCached(): Promise<boolean> {
  try {
    const cache = await caches.open(MODEL_CACHE);
    const manifestResponse = await cache.match(MANIFEST_URL);
    if (!manifestResponse) return false;
    const manifest = await manifestResponse.json() as Manifest;
    for (const file of manifest.files) {
      if (!(await cache.match(new URL(file.path, manifest.baseUrl).href))) return false;
      for (const external of file.externalData ?? []) {
        if (!(await cache.match(new URL(external.path, manifest.baseUrl).href))) return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}

export async function downloadAndVerifyModel(
  onProgress: (status: LocalModelStatus) => void,
) {
  const manifestResponse = await fetch(MANIFEST_URL);
  if (!manifestResponse.ok) throw new Error("Could not download the model manifest.");
  const manifest = await manifestResponse.json() as Manifest;
  const cache = await caches.open(MODEL_CACHE);
  await cache.put(MANIFEST_URL, manifestResponse.clone());

  const downloads: Array<{ url: string; bytes: number; sha: string }> = [];
  for (const file of manifest.files) {
    downloads.push({ url: new URL(file.path, manifest.baseUrl).href, bytes: file.bytes, sha: file.sha256 });
    for (const external of file.externalData ?? []) {
      downloads.push({ url: new URL(external.path, manifest.baseUrl).href, bytes: external.bytes, sha: external.sha256 });
    }
  }
  const tokenizerUrl = new URL(manifest.tokenizer ?? "tokenizer.json", manifest.baseUrl).href;
  downloads.push({ url: tokenizerUrl, bytes: manifest.tokenizerBytes ?? 0, sha: "" });

  const totalBytes = downloads.reduce((sum, item) => sum + item.bytes, 0);
  let downloadedBytes = 0;
  const startedAt = performance.now();

  for (const item of downloads) {
    const cached = await cache.match(item.url);
    if (cached) {
      downloadedBytes += item.bytes;
      onProgress({
        downloaded: false,
        downloading: true,
        totalBytes,
        downloadedBytes,
        percent: totalBytes ? Math.round(downloadedBytes * 100 / totalBytes) : 0,
        speedBytesPerSecond: 0,
        error: "",
      });
      continue;
    }

    const response = await fetch(item.url);
    if (!response.ok) throw new Error(`Could not download ${item.url}`);
    const bytes = await response.arrayBuffer();
    if (item.sha) {
      const actual = await sha256Hex(bytes);
      if (actual !== item.sha) throw new Error(`Checksum mismatch for ${item.url}`);
    }
    await cache.put(item.url, new Response(bytes, { headers: { "Content-Type": "application/octet-stream" } }));
    downloadedBytes += item.bytes;
    const elapsedSeconds = Math.max((performance.now() - startedAt) / 1000, 0.001);
    onProgress({
      downloaded: false,
      downloading: true,
      totalBytes,
      downloadedBytes,
      percent: totalBytes ? Math.round(downloadedBytes * 100 / totalBytes) : 0,
      speedBytesPerSecond: Math.round(downloadedBytes / elapsedSeconds),
      error: "",
    });
  }

  onProgress({
    downloaded: true,
    downloading: false,
    totalBytes,
    downloadedBytes: totalBytes,
    percent: 100,
    speedBytesPerSecond: 0,
    error: "",
  });
}

export async function deleteCachedModel() {
  await caches.delete(MODEL_CACHE);
}
