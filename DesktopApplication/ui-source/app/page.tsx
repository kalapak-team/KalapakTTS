"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type GpuStatus = {
  available: boolean;
  cuda_available: boolean;
  gpu_name: string | null;
  free_mib: number | null;
  total_mib: number | null;
  nvidia_smi_free_mib: number | null;
  torch_free_before_context_mib: number | null;
  torch_free_after_context_mib: number | null;
  torch_context_init_delta_mib: number | null;
  nvidia_smi_free_after_context_mib: number | null;
  nvidia_context_init_delta_mib: number | null;
  nvidia_torch_delta_mib: number | null;
  context_overhead_mib: number | null;
  model_peak_reserved_mib: number | null;
  model_need_mib: number | null;
  minimum_free_mib: number | null;
  safe_to_load_model: boolean;
  warning: string;
  checked_at: string;
};

type EngineStatus = {
  model_available: boolean;
  model_error: string | null;
  model_loaded: boolean;
  model_loading: boolean;
  model_device: string | null;
  selected_device: string;
  model_dtype: string | null;
  working_set_bytes: number | null;
  peak_ram_bytes: number | null;
  vram_used_bytes: number;
  vram_total_bytes: number;
  peak_vram_bytes: number;
  peak_vram_reserved_bytes: number;
  model_load_seconds: number | null;
  fallback_reason: string | null;
};

type StorageStatus = {
  status: "ready" | "recovery_required";
  root?: string;
  media_folder?: string;
  audio_storage_bytes?: number;
  generation_count?: number | null;
  schema_version?: number;
  backup_available: boolean;
  latest_backup?: string | null;
  message?: string;
};

type HistoryItem = {
  id: string;
  created_at: string;
  title: string | null;
  text: string;
  voice_id: string;
  pace: string;
  steps: number;
  speed: number;
  audio_duration: number | null;
  audio_missing: boolean;
  is_favorite: boolean;
  audio_url: string | null;
  srt_url: string;
  vtt_url: string;
};

type HistoryFilters = {
  search: string;
  date_from: string;
  date_to: string;
  voice: string;
  pace: string;
  steps: string;
  favorites_only: boolean;
};

type DraftSettings = {
  voice: string;
  pace: string;
  steps: number;
  speed: number;
  language: "km";
};

type GenerationResult = {
  id: string;
  duration_seconds: number;
  audio_url: string;
  srt_url: string;
  vtt_url: string;
  audio_sha256: string;
};

const initialSettings: DraftSettings = {
  voice: "vanna",
  pace: "natural",
  steps: 4,
  speed: 1,
  language: "km",
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "An unexpected local service error occurred.";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 ** 2) return `${bytes.toLocaleString()} B`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}

function localDateInput(daysAgo = 0): string {
  const date = new Date();
  date.setDate(date.getDate() - daysAgo);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const token = new URLSearchParams(window.location.hash.slice(1)).get("token");
  if (!token) throw new Error("The desktop API token is missing. Restart KalapakTTS Desktop.");
  const headers = new Headers(init.headers);
  headers.set("X-Desktop-Token", token);
  return fetch(input, { ...init, headers });
}

export default function Home() {
  const [text, setText] = useState("");
  const [settings, setSettings] = useState<DraftSettings>(initialSettings);
  const [startupGpuStatus, setStartupGpuStatus] = useState<GpuStatus | null>(null);
  const [gpuStatus, setGpuStatus] = useState<GpuStatus | null>(null);
  const [engineStatus, setEngineStatus] = useState<EngineStatus | null>(null);
  const [storageStatus, setStorageStatus] = useState<StorageStatus | null>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [savingDraft, setSavingDraft] = useState(false);
  const [draftLoaded, setDraftLoaded] = useState(false);
  const [draftSaveState, setDraftSaveState] = useState("");
  const [restoringBackup, setRestoringBackup] = useState(false);
  const [backupBusy, setBackupBusy] = useState(false);
  const [loadingDevice, setLoadingDevice] = useState<"cuda" | "cpu" | null>(null);
  const [generating, setGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState("");
  const [generation, setGeneration] = useState<GenerationResult | null>(null);
  const [chunkPreviews, setChunkPreviews] = useState<{ text: string; url: string }[]>([]);
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [historyHasMore, setHistoryHasMore] = useState(false);
  const [historyFilters, setHistoryFilters] = useState<HistoryFilters>({
    search: "",
    date_from: "",
    date_to: "",
    voice: "",
    pace: "",
    steps: "",
    favorites_only: false,
  });
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [mediaFolderPath, setMediaFolderPath] = useState("");
  const [generationAudioUrl, setGenerationAudioUrl] = useState<string | null>(null);
  const [historyAudioUrls, setHistoryAudioUrls] = useState<Record<string, string>>({});
  const [loadingAudioId, setLoadingAudioId] = useState<string | null>(null);
  const [movingMedia, setMovingMedia] = useState(false);
  const [cleanupBefore, setCleanupBefore] = useState(localDateInput(30));
  const [cleaningHistory, setCleaningHistory] = useState(false);
  const previewUrls = useRef<string[]>([]);
  const protectedObjectUrls = useRef<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const savedDraftSnapshot = useRef("");

  const createProtectedObjectUrl = async (path: string): Promise<string> => {
    const response = await apiFetch(path, { cache: "no-store" });
    if (!response.ok) throw new Error(`Local media could not be loaded (HTTP ${response.status}).`);
    const objectUrl = URL.createObjectURL(await response.blob());
    protectedObjectUrls.current.push(objectUrl);
    return objectUrl;
  };

  const downloadProtectedFile = async (path: string, fileName: string) => {
    try {
      const objectUrl = await createProtectedObjectUrl(path);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = fileName;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (error) {
      setStatusMessage(errorMessage(error));
    }
  };

  const refreshGpuStatus = useCallback(async () => {
    const response = await apiFetch("/api/system/gpu", { cache: "no-store" });
    if (!response.ok) throw new Error(`GPU status check failed (HTTP ${response.status}).`);
    const payload = await response.json() as { startup: GpuStatus; current: GpuStatus };
    setStartupGpuStatus((previous) => previous ?? payload.startup);
    setGpuStatus(payload.current);
  }, []);

  const refreshStorageStatus = useCallback(async (restoreDraft = false) => {
    const response = await apiFetch("/api/storage/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Local storage check failed (HTTP ${response.status}).`);
    const status = await response.json() as StorageStatus;
    setStorageStatus(status);
    setMediaFolderPath(status.media_folder ?? "");
    if (restoreDraft && status.status === "ready") {
      const draftResponse = await apiFetch("/api/storage/drafts/current", { cache: "no-store" });
      if (draftResponse.ok) {
        const draft = await draftResponse.json() as { text: string; settings: DraftSettings };
        setText(draft.text);
        setSettings({ ...initialSettings, ...draft.settings });
        savedDraftSnapshot.current = JSON.stringify({
          text: draft.text,
          settings: { ...initialSettings, ...draft.settings },
        });
        setDraftLoaded(true);
      } else if (draftResponse.status !== 404) {
        throw new Error(`Local draft could not be loaded (HTTP ${draftResponse.status}).`);
      } else {
        setDraftLoaded(true);
      }
    }
  }, []);

  const refreshHistory = useCallback(async (
    filters: HistoryFilters,
    offset = 0,
    append = false,
  ) => {
    const query = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (typeof value === "boolean") {
        if (value) query.set(key, "true");
      } else if (value) {
        query.set(key, value);
      }
    });
    query.set("limit", "100");
    query.set("offset", String(offset));
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const response = await apiFetch(`/api/tts/generations?${query.toString()}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`History could not be loaded (HTTP ${response.status}).`);
      const result = await response.json() as { items: HistoryItem[] };
      setHistoryItems((previous) => append ? [...previous, ...result.items] : result.items);
      setHistoryHasMore(result.items.length === 100);
    } catch (error) {
      setHistoryError(errorMessage(error));
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const refreshEngineStatus = useCallback(async () => {
    const response = await apiFetch("/api/tts/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`TTS engine status check failed (HTTP ${response.status}).`);
    setEngineStatus(await response.json() as EngineStatus);
  }, []);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        await refreshGpuStatus();
        await refreshStorageStatus(true);
        await refreshEngineStatus();
        if (active) setStatusMessage("");
      } catch (error) {
        if (active) setStatusMessage(errorMessage(error));
      }
    };

    void refresh();
    const intervalId = window.setInterval(() => {
      void refreshGpuStatus().catch((error: unknown) => {
        if (active) setStatusMessage(errorMessage(error));
      });
    }, 30_000);

    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, [refreshEngineStatus, refreshGpuStatus, refreshStorageStatus]);

  useEffect(() => {
    if (storageStatus?.status !== "ready") return;
    const timeoutId = window.setTimeout(() => {
      void refreshHistory(historyFilters);
    }, 250);
    return () => window.clearTimeout(timeoutId);
  }, [historyFilters, refreshHistory, storageStatus?.status]);

  useEffect(() => {
    if (!draftLoaded || storageStatus?.status !== "ready") return;
    const snapshot = JSON.stringify({ text, settings });
    if (snapshot === savedDraftSnapshot.current) return;
    const timeoutId = window.setTimeout(() => {
      setDraftSaveState("Saving draft…");
      void apiFetch("/api/storage/drafts/current", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: snapshot,
      }).then((response) => {
        if (!response.ok) throw new Error(`Draft autosave failed (HTTP ${response.status}).`);
        savedDraftSnapshot.current = snapshot;
        setDraftSaveState("Draft autosaved on this PC.");
      }).catch((error: unknown) => {
        setDraftSaveState("");
        setStatusMessage(errorMessage(error));
      });
    }, 1800);
    return () => window.clearTimeout(timeoutId);
  }, [draftLoaded, settings, storageStatus?.status, text]);

  useEffect(() => {
    const saveOnClose = () => {
      if (!draftLoaded || storageStatus?.status !== "ready") return;
      const snapshot = JSON.stringify({ text, settings });
      void apiFetch("/api/storage/drafts/current", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: snapshot,
        keepalive: true,
      }).catch(() => {
        console.error("Could not save the local draft while closing.");
      });
    };
    window.addEventListener("pagehide", saveOnClose);
    return () => window.removeEventListener("pagehide", saveOnClose);
  }, [draftLoaded, settings, storageStatus?.status, text]);

  useEffect(() => () => {
    previewUrls.current.forEach((url) => URL.revokeObjectURL(url));
    protectedObjectUrls.current.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  const saveDraft = async () => {
    setSavingDraft(true);
    setStatusMessage("");
    try {
      const response = await apiFetch("/api/storage/drafts/current", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, settings }),
      });
      if (!response.ok) throw new Error(`Draft could not be saved (HTTP ${response.status}).`);
      savedDraftSnapshot.current = JSON.stringify({ text, settings });
      setDraftSaveState("Draft saved on this PC.");
      setStatusMessage("Draft saved on this PC.");
    } catch (error) {
      setStatusMessage(errorMessage(error));
    } finally {
      setSavingDraft(false);
    }
  };

  const loadHistoryItem = async (item: HistoryItem) => {
    const nextSettings = {
      ...settings,
      voice: item.voice_id,
      pace: item.pace,
      steps: item.steps,
      speed: item.speed,
    };
    setText(item.text);
    setSettings(nextSettings);
    setStatusMessage(item.audio_missing ? "Audio file missing. The text and settings are ready to regenerate." : "History loaded into the editor.");
    setGeneration(
      item.audio_missing || !item.audio_url
        ? null
        : {
            id: item.id,
            duration_seconds: item.audio_duration ?? 0,
            audio_url: item.audio_url,
            srt_url: item.srt_url,
            vtt_url: item.vtt_url,
            audio_sha256: "",
          },
    );
    setGenerationAudioUrl(null);
    if (!item.audio_missing && item.audio_url) {
      try {
        setGenerationAudioUrl(await createProtectedObjectUrl(item.audio_url));
      } catch (error) {
        setStatusMessage(errorMessage(error));
      }
    }
    setGenerationProgress("");
    setChunkPreviews([]);
  };

  const loadHistoryAudio = async (item: HistoryItem) => {
    if (!item.audio_url || historyAudioUrls[item.id]) return;
    setLoadingAudioId(item.id);
    try {
      const objectUrl = await createProtectedObjectUrl(item.audio_url);
      setHistoryAudioUrls((previous) => ({ ...previous, [item.id]: objectUrl }));
    } catch (error) {
      setHistoryError(errorMessage(error));
    } finally {
      setLoadingAudioId(null);
    }
  };

  const updateHistoryItem = async (
    item: HistoryItem,
    action: "favorite" | "rename" | "delete" | "duplicate",
  ) => {
    if (action === "delete" && !window.confirm(`Delete “${item.title ?? item.text.slice(0, 60)}” and its local audio? This cannot be undone.`)) {
      return;
    }
    const title = action === "rename"
      ? window.prompt("Rename this history item:", item.title ?? item.text.slice(0, 80))
      : null;
    if (action === "rename" && title === null) return;
    const url = action === "favorite"
      ? `/api/tts/generations/${item.id}/favorite`
      : action === "duplicate"
        ? `/api/tts/generations/${item.id}/duplicate`
        : `/api/tts/generations/${item.id}`;
    const method = action === "favorite" ? "PUT" : action === "duplicate" ? "POST" : action === "delete" ? "DELETE" : "PATCH";
    try {
      const response = await apiFetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: action === "favorite"
          ? JSON.stringify({ favorite: !item.is_favorite })
          : action === "rename"
            ? JSON.stringify({ title })
            : undefined,
      });
      const result = await response.json() as { detail?: string; id?: string };
      if (!response.ok) throw new Error(result.detail ?? `History update failed (HTTP ${response.status}).`);
      if (action === "duplicate") {
        setText(item.text);
        setSettings({
          ...settings,
          voice: item.voice_id,
          pace: item.pace,
          steps: item.steps,
          speed: item.speed,
        });
        setStatusMessage("A duplicate history item was created.");
      } else if (action === "delete") {
        setStatusMessage("History item and its audio were deleted.");
      } else if (action === "rename") {
        setStatusMessage("History item renamed.");
      }
      await Promise.all([refreshHistory(historyFilters), refreshStorageStatus()]);
    } catch (error) {
      setHistoryError(errorMessage(error));
    }
  };

  const moveMediaFolder = async () => {
    if (!mediaFolderPath.trim()) {
      setStatusMessage("Enter a folder path for generated audio.");
      return;
    }
    if (!window.confirm(`Safely move generated audio to “${mediaFolderPath}”? Existing history will be updated.`)) return;
    setMovingMedia(true);
    try {
      const response = await apiFetch("/api/storage/media-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: mediaFolderPath }),
      });
      const result = await response.json() as { detail?: string; media_folder?: string; moved_files?: number };
      if (!response.ok) throw new Error(result.detail ?? `Media folder change failed (HTTP ${response.status}).`);
      setMediaFolderPath(result.media_folder ?? mediaFolderPath);
      setStatusMessage(`Media folder updated; ${result.moved_files ?? 0} file(s) moved.`);
      await refreshStorageStatus(true);
    } catch (error) {
      setStatusMessage(errorMessage(error));
    } finally {
      setMovingMedia(false);
    }
  };

  const cleanOldHistory = async () => {
    if (!cleanupBefore || !window.confirm(`Delete history and audio created before ${cleanupBefore}? This cannot be undone.`)) return;
    setCleaningHistory(true);
    try {
      const response = await apiFetch("/api/storage/cleanup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ before: cleanupBefore }),
      });
      const result = await response.json() as { detail?: string; deleted_generations?: number };
      if (!response.ok) throw new Error(result.detail ?? `History cleanup failed (HTTP ${response.status}).`);
      setStatusMessage(`Removed ${result.deleted_generations ?? 0} old history item(s).`);
      await Promise.all([refreshHistory(historyFilters), refreshStorageStatus()]);
    } catch (error) {
      setStatusMessage(errorMessage(error));
    } finally {
      setCleaningHistory(false);
    }
  };

  const restoreBackup = async () => {
    const backupName = storageStatus?.latest_backup ?? "the latest available backup";
    if (!window.confirm(`Restore ${backupName}? The current database will be replaced by this backup.`)) return;

    setRestoringBackup(true);
    setStatusMessage("");
    try {
      const response = await apiFetch("/api/storage/restore", { method: "POST" });
      const result = await response.json() as { detail?: string };
      if (!response.ok) throw new Error(result.detail ?? `Database restore failed (HTTP ${response.status}).`);
      await refreshStorageStatus();
      setStatusMessage("The last database backup was restored.");
    } catch (error) {
      setStatusMessage(errorMessage(error));
    } finally {
      setRestoringBackup(false);
    }
  };

  const exportBackup = async () => {
    setBackupBusy(true);
    setStatusMessage("");
    try {
      const response = await apiFetch("/api/storage/export", { cache: "no-store" });
      if (!response.ok) {
        const result = await response.json() as { detail?: string };
        throw new Error(result.detail ?? `Backup export failed (HTTP ${response.status}).`);
      }
      const objectUrl = URL.createObjectURL(await response.blob());
      protectedObjectUrls.current.push(objectUrl);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = "KalapakTTS-backup.zip";
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
      setStatusMessage("A checksummed local backup was exported.");
    } catch (error) {
      setStatusMessage(errorMessage(error));
    } finally {
      setBackupBusy(false);
    }
  };

  const importBackup = async (file: File) => {
    const importWarning = storageStatus?.status === "recovery_required"
      ? "The current database already needs recovery; it will be replaced by the selected backup."
      : "A recovery ZIP of the current profile will be kept first.";
    if (
      !window.confirm(
        `Import “${file.name}” and replace the current history, drafts and settings? ${importWarning}`,
      )
    ) {
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    setBackupBusy(true);
    setStatusMessage("");
    try {
      const response = await apiFetch("/api/storage/import", {
        method: "PUT",
        headers: { "Content-Type": "application/zip" },
        body: file,
      });
      const result = await response.json() as {
        detail?: string;
        generation_count?: number;
        audio_files?: number;
        missing_audio?: number;
      };
      if (!response.ok) throw new Error(result.detail ?? `Backup import failed (HTTP ${response.status}).`);
      await Promise.all([
        refreshStorageStatus(true),
        refreshHistory(historyFilters),
      ]);
      setStatusMessage(
        `Backup restored: ${result.generation_count ?? 0} history item(s), ${result.audio_files ?? 0} audio file(s), ${result.missing_audio ?? 0} missing audio file(s).`,
      );
    } catch (error) {
      setStatusMessage(errorMessage(error));
    } finally {
      setBackupBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const loadEngine = async (device: "cuda" | "cpu", tryGpuAnyway = false) => {
    if (
      device === "cpu"
      && !window.confirm(
        "CPU fallback is much slower. The measured short sample took 50.88 seconds; longer speech can take much longer. Continue loading the CPU model?",
      )
    ) {
      return;
    }
    setLoadingDevice(device);
    setStatusMessage(
      device === "cpu"
        ? "Loading VoxCPM2 on the CPU…"
        : tryGpuAnyway
          ? "Trying the GPU despite the VRAM warning…"
          : "Loading VoxCPM2 on the RTX 4050…",
    );
    try {
      const response = await apiFetch("/api/engine/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device, try_gpu_anyway: tryGpuAnyway }),
      });
      const result = await response.json() as EngineStatus & { detail?: string };
      if (!response.ok) throw new Error(result.detail ?? `Engine load failed (HTTP ${response.status}).`);
      setEngineStatus(result);
      setStatusMessage(`VoxCPM2 is loaded on ${result.model_device}.`);
    } catch (error) {
      setStatusMessage(errorMessage(error));
      try {
        await refreshEngineStatus();
      } catch (statusError) {
        setStatusMessage(
          `${errorMessage(error)} Engine status could not be refreshed: ${errorMessage(statusError)}`,
        );
      }
    } finally {
      setLoadingDevice(null);
    }
  };

  const generateSpeech = async () => {
    setGenerating(true);
    setGeneration(null);
    setGenerationAudioUrl(null);
    setGenerationProgress("Starting the local speech job…");
    setStatusMessage("");
    previewUrls.current.forEach((url) => URL.revokeObjectURL(url));
    previewUrls.current = [];
    setChunkPreviews([]);
    let completed = false;
    try {
      const response = await apiFetch("/api/tts/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          voice: settings.voice,
          pace: settings.pace,
          steps: settings.steps,
          speed: settings.speed,
          pronunciation_dictionary: true,
        }),
      });
      if (!response.ok) {
        const error = await response.json() as { detail?: string };
        throw new Error(error.detail ?? `Speech generation failed (HTTP ${response.status}).`);
      }
      if (!response.body) throw new Error("The local speech stream was not available.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finished = false;
      while (!finished) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        let boundary = buffer.indexOf("\n\n");
        while (boundary >= 0) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const eventName = block.split("\n").find((line) => line.startsWith("event:"))?.slice(6).trim();
          const dataLine = block.split("\n").find((line) => line.startsWith("data:"))?.slice(5).trim();
          if (eventName && dataLine) {
            const eventData = JSON.parse(dataLine) as Record<string, unknown>;
            if (eventName === "progress") {
              const current = Number(eventData.current_chunk ?? 0);
              const total = Number(eventData.total_chunks ?? 0);
              const eta = Number(eventData.estimated_seconds_remaining);
              const etaLabel = Number.isFinite(eta) && eta > 0 ? ` · about ${Math.ceil(eta)}s left` : "";
              const queuePosition = Number(eventData.queued_position ?? 0);
              const queueLabel = eventData.status === "queued" && queuePosition > 0
                ? `Queue position ${queuePosition} · `
                : "";
              setGenerationProgress(total
                ? `${queueLabel}Chunk ${current} of ${total}${etaLabel} · ${String(eventData.message ?? "")}`
                : `${queueLabel}${String(eventData.message ?? "Queued")}`);
            } else if (eventName === "chunk") {
              const base64Audio = String(eventData.audio_base64 ?? "");
              const bytes = Uint8Array.from(atob(base64Audio), (character) => character.charCodeAt(0));
              const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
              previewUrls.current.push(url);
              setChunkPreviews((previous) => [...previous, { text: String(eventData.text ?? ""), url }]);
              setGenerationProgress(`Chunk ${String(eventData.current_chunk)} of ${String(eventData.total_chunks)} ready`);
            } else if (eventName === "fallback") {
              setStatusMessage(
                "CUDA ran out of memory. Retrying on CPU, which is much slower; the measured short sample took 50.88 seconds.",
              );
              setGenerationProgress(
                String(eventData.message ?? "CUDA memory was exhausted; this chunk is being retried on CPU."),
              );
            } else if (eventName === "complete") {
              const result: GenerationResult = {
                id: String(eventData.id ?? ""),
                duration_seconds: Number(eventData.duration_seconds),
                audio_url: String(eventData.audio_url ?? ""),
                srt_url: String(eventData.srt_url ?? ""),
                vtt_url: String(eventData.vtt_url ?? ""),
                audio_sha256: String(eventData.audio_sha256 ?? ""),
              };
              if (!result.id || !result.audio_url || !result.srt_url || !result.vtt_url) {
                throw new Error("The local speech result was incomplete.");
              }
              completed = true;
              setGeneration(result);
              setGenerationAudioUrl(await createProtectedObjectUrl(result.audio_url));
              setGenerationProgress(`Speech ready · ${result.duration_seconds.toFixed(1)} seconds`);
              await Promise.all([
                refreshEngineStatus(),
                refreshStorageStatus(),
                refreshHistory(historyFilters),
              ]);
            } else if (eventName === "error") {
              throw new Error(String(eventData.error ?? "The local speech job failed."));
            }
          }
          boundary = buffer.indexOf("\n\n");
        }
        finished = done;
      }
      if (!completed) {
        setGenerationProgress("The local stream ended before speech was saved.");
      }
    } catch (error) {
      setStatusMessage(errorMessage(error));
      setGenerationProgress("");
      try {
        await refreshEngineStatus();
      } catch (statusError) {
        setStatusMessage(
          `${errorMessage(error)} Engine status could not be refreshed: ${errorMessage(statusError)}`,
        );
      }
    } finally {
      setGenerating(false);
    }
  };

  const lowVram = !engineStatus?.model_loaded
    && (!gpuStatus?.available || !gpuStatus?.cuda_available || !gpuStatus.safe_to_load_model);
  const gpuMessage = engineStatus?.model_device === "cuda"
    ? `VoxCPM2 is using the GPU. Keep the web TTS model stopped; ${gpuStatus?.free_mib ?? "—"} MiB is free. Other GPU apps reduce headroom and can trigger CPU fallback.`
    : engineStatus?.model_device === "cpu"
      ? "VoxCPM2 is using the much slower CPU fallback. The measured short sample took 50.88 seconds. Do not start another VoxCPM2 model in the web server."
    : gpuStatus?.warning
      ?? "Checking NVIDIA memory. Do not load VoxCPM2 until this check completes.";

  return (
    <main className="desktop-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">ក</span>
          <span><strong>KalapakTTS</strong><small>Khmer voice studio</small></span>
        </div>
        <div className="profile-status">
          <span className="status-dot" aria-hidden="true" />
          Local profile · This PC
        </div>
      </header>

      <section className={`gpu-banner ${lowVram ? "warning" : "ready"}`} role={lowVram ? "alert" : "status"} aria-live="polite">
        <span className="banner-icon" aria-hidden="true">{lowVram ? "!" : "✓"}</span>
        <div>
          <strong>
            {engineStatus?.model_device === "cuda"
              ? "VoxCPM2 is loaded on the GPU"
              : engineStatus?.model_device === "cpu"
                ? "VoxCPM2 is running on CPU fallback"
              : lowVram
                ? "GPU memory warning — choose CPU or explicitly try the GPU"
                : "GPU memory check complete"}
          </strong>
          <p>{gpuMessage}</p>
        </div>
        {gpuStatus?.available && (
          <span className="vram-readout">
            Start nvidia-smi: {startupGpuStatus?.nvidia_smi_free_mib ?? "—"} MiB ·
            CUDA now: {gpuStatus.free_mib ?? "—"} MiB ·
            nvidia-smi now: {gpuStatus.nvidia_smi_free_mib ?? "—"} MiB
            {` · nvidia/Torch gap: ${gpuStatus.nvidia_torch_delta_mib ?? "—"} MiB`}
            {typeof startupGpuStatus?.nvidia_context_init_delta_mib === "number"
              ? ` · context init Δ: ${startupGpuStatus.nvidia_context_init_delta_mib} MiB`
              : " · context init Δ: not measured"}
            {gpuStatus.model_need_mib === null
              ? " · Model need: not measured"
              : ` · Model need: ${gpuStatus.model_need_mib} MiB`}
          </span>
        )}
      </section>

      {storageStatus?.status === "recovery_required" && (
        <section className="recovery-banner" role="alert">
          <div>
            <strong>Local database recovery is required</strong>
            <p>{storageStatus.message}</p>
            {storageStatus.backup_available && <small>Available backup: {storageStatus.latest_backup}</small>}
          </div>
          {storageStatus.backup_available && (
            <button type="button" onClick={() => void restoreBackup()} disabled={restoringBackup}>
              {restoringBackup ? "Restoring…" : "Restore last backup"}
            </button>
          )}
        </section>
      )}

      <div className="workspace">
        <section className="editor-card" aria-labelledby="editor-title">
          <div className="section-heading">
            <div>
              <span className="eyebrow">YOUR PRIVATE WORKSPACE</span>
              <h1 id="editor-title">Khmer text to speech</h1>
              <p>Write and prepare speech on this PC. No account is required.</p>
            </div>
            <span className="offline-badge"><span aria-hidden="true" />Local app</span>
          </div>

          <label className="field-label" htmlFor="speech-text">Text · អត្ថបទ</label>
          <textarea
            id="speech-text"
            maxLength={5000}
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="សូមបញ្ចូលអត្ថបទភាសាខ្មែររបស់អ្នក…"
          />
          <div className="character-count">{text.length.toLocaleString()} / 5,000</div>

          <div className="settings-grid">
            <label>
              <span className="field-label">Voice · សំឡេង</span>
              <select value={settings.voice} onChange={(event) => setSettings({ ...settings, voice: event.target.value })}>
                <option value="vanna">Vanna</option>
              </select>
            </label>
            <label>
              <span className="field-label">Language · ភាសា</span>
              <select value={settings.language} disabled>
                <option value="km">Khmer · ភាសាខ្មែរ</option>
              </select>
            </label>
            <label>
              <span className="field-label">Pace · ល្បឿន</span>
              <select value={settings.pace} onChange={(event) => setSettings({ ...settings, pace: event.target.value })}>
                <option value="relaxed">Relaxed · 12 CPS</option>
                <option value="natural">Natural · 14 CPS</option>
                <option value="brisk">Brisk · 16 CPS</option>
              </select>
            </label>
            <label>
              <span className="field-label">Steps</span>
              <select value={settings.steps} onChange={(event) => setSettings({ ...settings, steps: Number(event.target.value) })}>
                <option value={4}>Fast · 4</option>
                <option value={8}>Balanced · 8</option>
                <option value={12}>Quality · 12</option>
              </select>
            </label>
            <label>
              <span className="field-label">Speed · pitch-preserving</span>
              <select value={settings.speed} onChange={(event) => setSettings({ ...settings, speed: Number(event.target.value) })}>
                <option value={0.85}>0.85× · Slower</option>
                <option value={1}>1.00× · Normal</option>
                <option value={1.15}>1.15× · Faster</option>
              </select>
            </label>
          </div>

          {!engineStatus?.model_loaded && (
            <div className="engine-actions">
              <button
                className="primary-button"
                type="button"
                onClick={() => void loadEngine("cuda")}
                disabled={
                  loadingDevice !== null
                  || !engineStatus?.model_available
                  || !gpuStatus?.available
                  || !gpuStatus.cuda_available
                  || !gpuStatus.safe_to_load_model
                  || storageStatus?.status !== "ready"
                }
              >
                {loadingDevice === "cuda" ? "Loading GPU model…" : "Load on RTX 4050"}
              </button>
              {gpuStatus?.available && gpuStatus.cuda_available && !gpuStatus.safe_to_load_model && (
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => void loadEngine("cuda", true)}
                  disabled={
                    loadingDevice !== null
                    || !engineStatus?.model_available
                    || storageStatus?.status !== "ready"
                  }
                >
                  {loadingDevice === "cuda" ? "Trying GPU…" : "Try GPU anyway"}
                </button>
              )}
              <button
                className="secondary-button"
                type="button"
                onClick={() => void loadEngine("cpu")}
                disabled={
                  loadingDevice !== null
                  || !engineStatus?.model_available
                  || storageStatus?.status !== "ready"
                }
              >
                {loadingDevice === "cpu" ? "Loading CPU model…" : "Use CPU fallback (much slower)"}
              </button>
            </div>
          )}

          <div className="editor-actions">
            <button
              className="primary-button"
              type="button"
              onClick={() => void generateSpeech()}
              disabled={
                generating
                || !text.trim()
                || !engineStatus?.model_loaded
                || storageStatus?.status !== "ready"
              }
            >
              {generating ? "Generating…" : "Generate speech"}
            </button>
            <button className="secondary-button" type="button" onClick={() => void saveDraft()} disabled={savingDraft || storageStatus?.status !== "ready"}>
              {savingDraft ? "Saving…" : "Save local draft"}
            </button>
          </div>
          <p className="engine-note">
            {engineStatus?.model_error
              ?? engineStatus?.fallback_reason
              ?? (engineStatus?.model_loaded
                ? `VoxCPM2 is loaded on ${engineStatus.model_device} in BF16 mode. Stop the web TTS model before using this desktop engine.`
                : "Load the local model once; it stays warm for this app session. CUDA is blocked below the startup VRAM threshold.")}
          </p>
          {generationProgress && <p className="inline-status" role="status">{generationProgress}</p>}
          {chunkPreviews.length > 0 && (
            <section className="preview-list" aria-label="Streaming audio chunks">
              <h2>Ready chunks</h2>
              {chunkPreviews.map((preview, index) => (
                <div className="preview-item" key={`${index}-${preview.url}`}>
                  <span>{preview.text}</span>
                  <audio controls preload="none" src={preview.url}>Audio playback is not supported.</audio>
                </div>
              ))}
            </section>
          )}
          {generation && (
            <section className="generation-result" aria-label="Generated speech">
              <h2>Generated audio · {generation.duration_seconds.toFixed(1)} seconds</h2>
              <audio controls preload="metadata" src={generationAudioUrl ?? undefined}>Audio playback is not supported.</audio>
              <div className="download-links">
                <button type="button" onClick={() => void downloadProtectedFile(generation.audio_url, `${generation.id}.wav`)}>Download WAV</button>
                <button type="button" onClick={() => void downloadProtectedFile(generation.srt_url, `${generation.id}.srt`)}>Download SRT</button>
                <button type="button" onClick={() => void downloadProtectedFile(generation.vtt_url, `${generation.id}.vtt`)}>Download VTT</button>
              </div>
            </section>
          )}
          {statusMessage && <p className="inline-status" role="status">{statusMessage}</p>}
        </section>

        <aside className="status-card" aria-label="Local app status">
          <span className="eyebrow">THIS PC</span>
          <h2>Local app status</h2>
          <dl>
            <div><dt>Profile</dt><dd>Automatic · no sign-in</dd></div>
            <div><dt>Storage</dt><dd>{storageStatus?.status === "ready" ? "SQLite ready" : storageStatus?.status === "recovery_required" ? "Recovery required" : "Checking…"}</dd></div>
            <div><dt>Database version</dt><dd>{storageStatus?.schema_version ?? "—"}</dd></div>
            <div><dt>Profile folder</dt><dd>{storageStatus?.root ?? "Loading local path…"}</dd></div>
            <div><dt>Generated audio</dt><dd>{formatBytes(storageStatus?.audio_storage_bytes ?? 0)}</dd></div>
            <div><dt>History items</dt><dd>{storageStatus?.generation_count ?? "—"}</dd></div>
            <div><dt>GPU</dt><dd>{gpuStatus?.gpu_name ?? (gpuStatus ? "Unavailable" : "Checking…")}</dd></div>
            <div><dt>Engine</dt><dd>{engineStatus?.model_loaded ? `${engineStatus.model_device} · ${engineStatus.model_dtype ?? "loaded"}` : "Not loaded"}</dd></div>
            <div>
              <dt>Peak VRAM allocated / reserved</dt>
              <dd>
                {engineStatus?.peak_vram_bytes
                  ? `${(engineStatus.peak_vram_bytes / 1024 ** 2).toFixed(0)} / ${
                      (engineStatus.peak_vram_reserved_bytes / 1024 ** 2).toFixed(0)
                    } MiB`
                  : "—"}
              </dd>
            </div>
            <div>
              <dt>RAM</dt>
              <dd>
                {engineStatus?.working_set_bytes
                  ? `${(engineStatus.working_set_bytes / 1024 ** 3).toFixed(1)} / ${
                      engineStatus.peak_ram_bytes
                        ? (engineStatus.peak_ram_bytes / 1024 ** 3).toFixed(1)
                        : "—"
                    } GiB (peak)`
                  : "—"}
              </dd>
            </div>
          </dl>
          <p className="privacy-note">Drafts and settings are stored on this PC. They are not uploaded to an account.</p>
        </aside>
      </div>

      <section className="history-card" aria-labelledby="history-title">
        <div className="history-heading">
          <div>
            <span className="eyebrow">LOCAL LIBRARY</span>
            <h2 id="history-title">Generation history</h2>
            <p>Search, filter, replay, rename, duplicate or remove speech saved on this PC.</p>
          </div>
          <strong>{storageStatus?.generation_count ?? historyItems.length} item(s) · {formatBytes(storageStatus?.audio_storage_bytes ?? 0)}</strong>
        </div>

        <div className="history-filters">
          <label className="history-search">
            <span className="field-label">Search history</span>
            <input
              type="search"
              value={historyFilters.search}
              onChange={(event) => setHistoryFilters({ ...historyFilters, search: event.target.value })}
              placeholder="Search title or text"
            />
          </label>
          <label>
            <span className="field-label">From</span>
            <input type="date" value={historyFilters.date_from} onChange={(event) => setHistoryFilters({ ...historyFilters, date_from: event.target.value })} />
          </label>
          <label>
            <span className="field-label">To</span>
            <input type="date" value={historyFilters.date_to} onChange={(event) => setHistoryFilters({ ...historyFilters, date_to: event.target.value })} />
          </label>
          <label>
            <span className="field-label">Voice</span>
            <select value={historyFilters.voice} onChange={(event) => setHistoryFilters({ ...historyFilters, voice: event.target.value })}>
              <option value="">All voices</option>
              <option value="vanna">Vanna</option>
            </select>
          </label>
          <label>
            <span className="field-label">Pace</span>
            <select value={historyFilters.pace} onChange={(event) => setHistoryFilters({ ...historyFilters, pace: event.target.value })}>
              <option value="">All paces</option>
              <option value="relaxed">Relaxed</option>
              <option value="natural">Natural</option>
              <option value="brisk">Brisk</option>
            </select>
          </label>
          <label>
            <span className="field-label">Steps</span>
            <select value={historyFilters.steps} onChange={(event) => setHistoryFilters({ ...historyFilters, steps: event.target.value })}>
              <option value="">All steps</option>
              <option value="4">4</option>
              <option value="8">8</option>
              <option value="12">12</option>
            </select>
          </label>
          <label className="favorite-filter">
            <input
              type="checkbox"
              checked={historyFilters.favorites_only}
              onChange={(event) => setHistoryFilters({ ...historyFilters, favorites_only: event.target.checked })}
            />
            Favorites only
          </label>
        </div>

        {historyLoading && <p className="inline-status" role="status">Loading local history…</p>}
        {historyError && <p className="inline-error" role="alert">{historyError}</p>}
        {!historyLoading && !historyError && historyItems.length === 0 && (
          <p className="history-empty">No saved generations match these filters.</p>
        )}
        <div className="history-list">
          {historyItems.map((item) => (
            <article className="history-item" key={item.id}>
              <div className="history-item-main">
                <div className="history-item-heading">
                  <h3>{item.title || item.text.slice(0, 80)}</h3>
                  <time dateTime={item.created_at}>{new Date(item.created_at).toLocaleString()}</time>
                </div>
                <p className="history-text">{item.text}</p>
                <small>
                  {item.voice_id} · {item.pace} · {item.steps} steps · {item.speed.toFixed(2)}×
                  {item.audio_duration ? ` · ${item.audio_duration.toFixed(1)} sec` : ""}
                </small>
                {item.audio_missing
                  ? <p className="missing-audio" role="alert">Audio file missing. Load the text and settings to regenerate.</p>
                  : historyAudioUrls[item.id]
                    ? <audio controls preload="none" src={historyAudioUrls[item.id]}>Audio playback is not supported.</audio>
                    : <button
                        type="button"
                        className="secondary-button"
                        onClick={() => void loadHistoryAudio(item)}
                        disabled={loadingAudioId === item.id}
                      >
                        {loadingAudioId === item.id ? "Loading audio…" : "Play audio"}
                      </button>}
              </div>
              <div className="history-actions">
                <button type="button" className="secondary-button" onClick={() => void loadHistoryItem(item)}>
                  {item.audio_missing ? "Regenerate missing audio" : "Open / regenerate"}
                </button>
                {!item.audio_missing && (
                  <>
                    <button type="button" className="history-icon-button" onClick={() => void downloadProtectedFile(item.audio_url ?? "", `${item.id}.wav`)}>Download WAV</button>
                    <button type="button" className="history-icon-button" onClick={() => void downloadProtectedFile(item.srt_url, `${item.id}.srt`)}>Download SRT</button>
                    <button type="button" className="history-icon-button" onClick={() => void downloadProtectedFile(item.vtt_url, `${item.id}.vtt`)}>Download VTT</button>
                  </>
                )}
                <button type="button" className="history-icon-button" onClick={() => void updateHistoryItem(item, "favorite")} aria-label={item.is_favorite ? "Remove favorite" : "Add favorite"}>
                  {item.is_favorite ? "★ Favorite" : "☆ Favorite"}
                </button>
                <button type="button" className="history-icon-button" onClick={() => void updateHistoryItem(item, "rename")}>Rename</button>
                <button type="button" className="history-icon-button" onClick={() => void updateHistoryItem(item, "duplicate")}>Duplicate</button>
                <button type="button" className="history-icon-button danger" onClick={() => void updateHistoryItem(item, "delete")}>Delete</button>
              </div>
            </article>
          ))}
        </div>
        {historyHasMore && (
          <button
            type="button"
            className="secondary-button load-more-button"
            onClick={() => void refreshHistory(historyFilters, historyItems.length, true)}
            disabled={historyLoading}
          >
            Load more history
          </button>
        )}
      </section>

      <section className="storage-tools" aria-label="Local media storage tools">
        <div>
          <span className="eyebrow">LOCAL STORAGE</span>
          <h2>Storage, backup and cleanup</h2>
          <p>Backups include local history, drafts, settings, voice files and generated audio with SHA-256 verification.</p>
        </div>
        <label className="media-folder-control">
          <span className="field-label">Media folder path</span>
          <input value={mediaFolderPath} onChange={(event) => setMediaFolderPath(event.target.value)} />
        </label>
        <button type="button" className="secondary-button" onClick={() => void moveMediaFolder()} disabled={movingMedia || storageStatus?.status !== "ready"}>
          {movingMedia ? "Moving audio…" : "Move media folder"}
        </button>
        <label>
          <span className="field-label">Clean history before</span>
          <input type="date" value={cleanupBefore} onChange={(event) => setCleanupBefore(event.target.value)} />
        </label>
        <button type="button" className="secondary-button cleanup-button" onClick={() => void cleanOldHistory()} disabled={cleaningHistory || storageStatus?.status !== "ready"}>
          {cleaningHistory ? "Cleaning…" : "Clean old history"}
        </button>
        <div className="backup-actions">
          <button type="button" className="secondary-button" onClick={() => void exportBackup()} disabled={backupBusy || storageStatus?.status !== "ready"}>
            {backupBusy ? "Preparing backup…" : "Export backup ZIP"}
          </button>
          <button type="button" className="secondary-button" onClick={() => fileInputRef.current?.click()} disabled={backupBusy}>
            {backupBusy ? "Restoring backup…" : "Import backup ZIP"}
          </button>
          <input
            ref={fileInputRef}
            className="backup-file-input"
            type="file"
            accept=".zip,application/zip"
            onChange={(event) => {
              const file = event.currentTarget.files?.[0];
              if (file) void importBackup(file);
            }}
          />
        </div>
      </section>
      {draftSaveState && <p className="draft-save-state" role="status">{draftSaveState}</p>}
    </main>
  );
}
