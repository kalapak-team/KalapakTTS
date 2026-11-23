"use client";

import Image from "next/image";
import { useCallback, useEffect, useRef, useState } from "react";

type GpuStatus = {
  available: boolean;
  cuda_available: boolean;
  gpu_name: string | null;
  driver_version: string | null;
  minimum_driver_version: string;
  driver_compatible: boolean | null;
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
  free_disk_bytes?: number;
  generation_count?: number | null;
  schema_version?: number;
  backup_available: boolean;
  latest_backup?: string | null;
  message?: string;
};

type AppLockStatus = {
  enabled: boolean;
  locked: boolean;
  lockout_remaining_seconds: number;
  max_failed_attempts: number;
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

function label(english: string, khmer: string, language: "en" | "km"): string {
  return language === "km" ? khmer : english;
}

function HistoryNotice({
  message,
  language,
  onDismiss,
}: {
  message: string;
  language: "en" | "km";
  onDismiss: () => void;
}) {
  return (
    <div className="history-notice" role="alert">
      <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="9" />
        <path d="M12 8v4m0 4h.01" />
      </svg>
      <p>{message}</p>
      <button
        type="button"
        onClick={onDismiss}
        aria-label={label("Dismiss notification", "បិទការជូនដំណឹង", language)}
        title={label("Dismiss", "បិទ", language)}
      >
        <svg aria-hidden="true" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <path d="m18 6-12 12M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}

function StartupBackdrop() {
  return (
    <div className="startup-backdrop" aria-hidden="true">
      <svg viewBox="0 0 1440 900" preserveAspectRatio="none" focusable="false">
        <defs>
          <linearGradient id="startup-wave-top" x1="860" y1="15" x2="1390" y2="205" gradientUnits="userSpaceOnUse">
            <stop stopColor="#21125f" />
            <stop offset=".56" stopColor="#5924d3" />
            <stop offset="1" stopColor="#8a3dff" />
          </linearGradient>
          <linearGradient id="startup-wave-bottom" x1="40" y1="758" x2="775" y2="900" gradientUnits="userSpaceOnUse">
            <stop stopColor="#6b35fa" />
            <stop offset=".5" stopColor="#27136f" />
            <stop offset="1" stopColor="#100d2b" />
          </linearGradient>
          <pattern id="startup-dots" width="20" height="20" patternUnits="userSpaceOnUse">
            <circle cx="3" cy="3" r="2" fill="#8064e8" fillOpacity=".42" />
          </pattern>
        </defs>
        <rect width="1440" height="900" fill="#090a18" />
        <rect x="0" y="24" width="240" height="170" fill="url(#startup-dots)" />
        <rect x="1210" y="615" width="230" height="250" fill="url(#startup-dots)" />
        <path
          d="M846 0h594v175c-88 55-162-8-251-34-103-30-175-1-246 31-52 23-112 17-153-9 68-40 95-95 56-163Z"
          fill="url(#startup-wave-top)"
          fillOpacity=".72"
        />
        <path
          d="M878 165c73-17 123-83 218-82 105 1 173 77 259 48 35-12 62-35 85-61"
          fill="none"
          stroke="#9b63ff"
          strokeOpacity=".66"
          strokeWidth="2"
        />
        <path
          d="M0 690c115-66 207-29 307 26 111 62 226 91 373 62l103 122H0V690Z"
          fill="url(#startup-wave-bottom)"
          fillOpacity=".9"
        />
        <path
          d="M0 708c123-68 207-31 305 21 112 60 228 83 370 53"
          fill="none"
          stroke="#884fff"
          strokeOpacity=".72"
          strokeWidth="2"
        />
        <path
          d="M0 736c119-62 202-24 296 25 105 55 222 78 349 56"
          fill="none"
          stroke="#4320b7"
          strokeOpacity=".8"
          strokeWidth="12"
        />
      </svg>
    </div>
  );
}

function MicrophoneIllustration() {
  return (
    <svg className="mic-svg" viewBox="0 0 240 240" role="presentation" focusable="false">
      <defs>
        <linearGradient id="mic-metal" x1="68" y1="34" x2="145" y2="148" gradientUnits="userSpaceOnUse">
          <stop stopColor="#fff" />
          <stop offset=".42" stopColor="#b8b8dc" />
          <stop offset=".72" stopColor="#77779f" />
          <stop offset="1" stopColor="#e9e9ff" />
        </linearGradient>
        <linearGradient id="mic-base" x1="66" y1="192" x2="160" y2="220" gradientUnits="userSpaceOnUse">
          <stop stopColor="#38217f" />
          <stop offset=".52" stopColor="#9b53ff" />
          <stop offset="1" stopColor="#302078" />
        </linearGradient>
        <linearGradient id="mic-wave" x1="158" y1="60" x2="222" y2="130" gradientUnits="userSpaceOnUse">
          <stop stopColor="#211544" />
          <stop offset="1" stopColor="#171632" />
        </linearGradient>
      </defs>
      <ellipse cx="113" cy="211" rx="76" ry="15" fill="#5121d2" fillOpacity=".32" />
      <ellipse cx="113" cy="206" rx="60" ry="12" fill="url(#mic-base)" stroke="#a568ff" strokeWidth="2" />
      <ellipse cx="113" cy="202" rx="43" ry="7" fill="#251553" stroke="#7144de" strokeWidth="2" />
      <ellipse cx="112" cy="144" rx="64" ry="19" fill="none" stroke="#8c4cff" strokeOpacity=".68" strokeWidth="2" transform="rotate(-13 112 144)" />
      <path d="M89 133v22c0 16 10 25 24 25s24-9 24-25v-22" fill="none" stroke="#d4d3f0" strokeWidth="8" strokeLinecap="round" />
      <path d="M113 178v19" stroke="#d8d8f4" strokeWidth="8" strokeLinecap="round" />
      <path d="M93 198h40" stroke="#d8d8f4" strokeWidth="7" strokeLinecap="round" />
      <rect x="78" y="25" width="70" height="120" rx="35" fill="url(#mic-metal)" stroke="#e9e8ff" strokeWidth="2" />
      <path d="M88 61h50M86 71h54M85 81h56M85 91h56M87 101h52" stroke="#3b3a5e" strokeOpacity=".88" strokeWidth="4" strokeLinecap="round" />
      <path d="M95 37h36" stroke="#fff" strokeOpacity=".82" strokeWidth="3" strokeLinecap="round" />
      <rect x="157" y="54" width="66" height="77" rx="14" fill="url(#mic-wave)" stroke="#7545d9" strokeWidth="2" />
      <path d="M168 92h5m6-13v26m8-36v46m9-35v24m9-31v39m8-24v9" stroke="#b56aff" strokeWidth="4" strokeLinecap="round" />
      <circle cx="57" cy="71" r="3" fill="#a96cff" />
      <circle cx="175" cy="40" r="2" fill="#c19aff" />
      <circle cx="204" cy="151" r="2" fill="#8f59ff" />
    </svg>
  );
}

export default function Home() {
  const [language, setLanguage] = useState<"en" | "km">("en");
  const [activeView, setActiveView] = useState<"generate" | "history">("generate");
  const [workspacePanel, setWorkspacePanel] = useState<"settings" | "history">("settings");
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [bootReady, setBootReady] = useState(false);
  const [bootProgress, setBootProgress] = useState("Connecting to the local service…");
  const [autoLoadPending, setAutoLoadPending] = useState(false);
  const [appLockStatus, setAppLockStatus] = useState<AppLockStatus | null>(null);
  const [lockPassword, setLockPassword] = useState("");
  const [lockConfirmation, setLockConfirmation] = useState("");
  const [unlockPassword, setUnlockPassword] = useState("");
  const [disablePassword, setDisablePassword] = useState("");
  const [lockBusy, setLockBusy] = useState(false);
  const [lockError, setLockError] = useState("");
  const [text, setText] = useState("");
  const [settings, setSettings] = useState<DraftSettings>(initialSettings);
  const [startupGpuStatus, setStartupGpuStatus] = useState<GpuStatus | null>(null);
  const [gpuStatus, setGpuStatus] = useState<GpuStatus | null>(null);
  const [engineStatus, setEngineStatus] = useState<EngineStatus | null>(null);
  const [storageStatus, setStorageStatus] = useState<StorageStatus | null>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [savingDraft, setSavingDraft] = useState(false);
  const [draftLoaded, setDraftLoaded] = useState(false);
  const [draftSaveState, setDraftSaveState] = useState<"saving" | "saved" | "autosaved" | "">("");
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
  const [playingHistoryId, setPlayingHistoryId] = useState<string | null>(null);
  const [movingMedia, setMovingMedia] = useState(false);
  const [cleanupBefore, setCleanupBefore] = useState(localDateInput(30));
  const [cleaningHistory, setCleaningHistory] = useState(false);
  const previewUrls = useRef<string[]>([]);
  const protectedObjectUrls = useRef<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textImportInputRef = useRef<HTMLInputElement>(null);
  const speechTextAreaRef = useRef<HTMLTextAreaElement>(null);
  const historyAudioPlayers = useRef<Record<string, HTMLAudioElement | null>>({});
  const savedDraftSnapshot = useRef("");

  const resizeSpeechTextArea = useCallback(() => {
    const textArea = speechTextAreaRef.current;
    if (!textArea) return;
    const maxHeight = Number.parseFloat(window.getComputedStyle(textArea).maxHeight);
    textArea.style.height = "auto";
    textArea.style.height = `${Math.min(
      textArea.scrollHeight,
      Number.isFinite(maxHeight) ? maxHeight : 480,
    )}px`;
  }, []);

  useEffect(() => {
    resizeSpeechTextArea();
  }, [text, bootReady, appLockStatus?.locked, resizeSpeechTextArea]);

  useEffect(() => {
    window.addEventListener("resize", resizeSpeechTextArea);
    return () => window.removeEventListener("resize", resizeSpeechTextArea);
  }, [resizeSpeechTextArea]);

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
    return payload.current;
  }, []);

  const refreshStorageStatus = useCallback(async (restoreDraft = false) => {
    const response = await apiFetch("/api/storage/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Local storage check failed (HTTP ${response.status}).`);
    const status = await response.json() as StorageStatus;
    setStorageStatus(status);
    setMediaFolderPath(status.media_folder ?? "");
    if (restoreDraft && status.status === "ready") {
      const languageResponse = await apiFetch("/api/storage/settings/ui_language", { cache: "no-store" });
      if (languageResponse.ok) {
        const savedLanguage = (await languageResponse.json() as { value: unknown }).value;
        if (savedLanguage === "en" || savedLanguage === "km") setLanguage(savedLanguage);
      } else if (languageResponse.status !== 404) {
        throw new Error(`Local language setting could not be loaded (HTTP ${languageResponse.status}).`);
      }
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
    return status;
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
    const status = await response.json() as EngineStatus;
    setEngineStatus(status);
    return status;
  }, []);

  const prepareWorkspace = useCallback(async (restoreDraft: boolean) => {
    const [gpu, storage, engine] = await Promise.all([
      refreshGpuStatus(),
      refreshStorageStatus(restoreDraft),
      refreshEngineStatus(),
    ]);
    if (engine.model_loaded) {
      setBootReady(true);
    } else if (
      engine.model_available
      && storage.status === "ready"
      && gpu.safe_to_load_model
    ) {
      setAutoLoadPending(true);
    } else {
      setBootReady(true);
    }
    return { gpu, storage, engine };
  }, [refreshEngineStatus, refreshGpuStatus, refreshStorageStatus]);

  useEffect(() => {
    let active = true;
    const bootstrap = async () => {
      try {
        const securityResponse = await apiFetch("/api/security/status", { cache: "no-store" });
        if (!securityResponse.ok) {
          throw new Error(`Local security check failed (HTTP ${securityResponse.status}).`);
        }
        const security = await securityResponse.json() as AppLockStatus;
        if (!active) return;
        setAppLockStatus(security);
        if (security.locked) {
          setBootReady(true);
          return;
        }
        setBootProgress("Checking the local model and available GPU memory…");
        await prepareWorkspace(true);
        if (!active) return;
        setStatusMessage("");
      } catch (error) {
        if (active) {
          setStatusMessage(errorMessage(error));
          setBootReady(true);
        }
      }
    };

    void bootstrap();
    return () => {
      active = false;
    };
  }, [prepareWorkspace]);

  useEffect(() => {
    if (!bootReady || appLockStatus?.locked) return;
    const intervalId = window.setInterval(() => {
      void refreshGpuStatus().catch((error: unknown) => {
        setStatusMessage(errorMessage(error));
      });
    }, 30_000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [appLockStatus?.locked, bootReady, refreshGpuStatus]);

  const autoLoadStarted = useRef(false);
  useEffect(() => {
    if (!autoLoadPending || autoLoadStarted.current) return;
    autoLoadStarted.current = true;
    void loadEngine("cuda").finally(() => {
      setAutoLoadPending(false);
      setBootReady(true);
    });
  }, [autoLoadPending]);

  useEffect(() => {
    if (appLockStatus?.locked || storageStatus?.status !== "ready") return;
    const timeoutId = window.setTimeout(() => {
      void refreshHistory(historyFilters);
    }, 250);
    return () => window.clearTimeout(timeoutId);
  }, [appLockStatus?.locked, historyFilters, refreshHistory, storageStatus?.status]);

  useEffect(() => {
    if (appLockStatus?.locked || !draftLoaded || storageStatus?.status !== "ready") return;
    const snapshot = JSON.stringify({ text, settings });
    if (snapshot === savedDraftSnapshot.current) return;
    setDraftSaveState("");
    const timeoutId = window.setTimeout(() => {
      setDraftSaveState("saving");
      void apiFetch("/api/storage/drafts/current", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: snapshot,
      }).then((response) => {
        if (!response.ok) throw new Error(`Draft autosave failed (HTTP ${response.status}).`);
        savedDraftSnapshot.current = snapshot;
        setDraftSaveState("autosaved");
      }).catch((error: unknown) => {
        setDraftSaveState("");
        setStatusMessage(errorMessage(error));
      });
    }, 1800);
    return () => window.clearTimeout(timeoutId);
  }, [appLockStatus?.locked, draftLoaded, settings, storageStatus?.status, text]);

  useEffect(() => {
    const saveOnClose = () => {
      if (appLockStatus?.locked || !draftLoaded || storageStatus?.status !== "ready") return;
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
  }, [appLockStatus?.locked, draftLoaded, settings, storageStatus?.status, text]);

  useEffect(() => () => {
    previewUrls.current.forEach((url) => URL.revokeObjectURL(url));
    protectedObjectUrls.current.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  const saveDraft = async () => {
    setSavingDraft(true);
    setDraftSaveState("saving");
    setStatusMessage("");
    try {
      const response = await apiFetch("/api/storage/drafts/current", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, settings }),
      });
      if (!response.ok) throw new Error(`Draft could not be saved (HTTP ${response.status}).`);
      savedDraftSnapshot.current = JSON.stringify({ text, settings });
      setDraftSaveState("saved");
    } catch (error) {
      setDraftSaveState("");
      setStatusMessage(errorMessage(error));
    } finally {
      setSavingDraft(false);
    }
  };

  const importTextFile = async (file: File) => {
    try {
      if (file.size > 100_000) {
        throw new Error(label(
          "The selected text file is too large to import.",
          "ឯកសារអត្ថបទដែលបានជ្រើសរើសធំពេកសម្រាប់នាំចូល។",
          language,
        ));
      }
      const importedText = (await file.text()).replace(/^\uFEFF/, "");
      if (importedText.length > 5000) {
        throw new Error(label(
          "Text files must contain no more than 5,000 characters.",
          "ឯកសារអត្ថបទមិនត្រូវមានលើសពី ៥,០០០ តួអក្សរទេ។",
          language,
        ));
      }
      setText(importedText);
      setStatusMessage(label(`Imported ${file.name}.`, `បាននាំចូល ${file.name}។`, language));
    } catch (error) {
      setStatusMessage(errorMessage(error));
    } finally {
      if (textImportInputRef.current) textImportInputRef.current.value = "";
    }
  };

  const stopHistoryAudio = () => {
    Object.values(historyAudioPlayers.current).forEach((audio) => audio?.pause());
    setPlayingHistoryId(null);
  };

  const loadHistoryItem = async (item: HistoryItem) => {
    stopHistoryAudio();
    setActiveView("generate");
    setWorkspacePanel("settings");
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

  const loadHistoryAudio = async (item: HistoryItem, autoplay = false) => {
    if (!item.audio_url) return;
    setHistoryError("");
    let objectUrl = historyAudioUrls[item.id];
    if (!objectUrl) {
      setLoadingAudioId(item.id);
      try {
        objectUrl = await createProtectedObjectUrl(item.audio_url);
        setHistoryAudioUrls((previous) => ({ ...previous, [item.id]: objectUrl }));
      } catch (error) {
        setHistoryError(errorMessage(error));
        return;
      } finally {
        setLoadingAudioId(null);
      }
    }

    if (!autoplay) return;
    const audio = historyAudioPlayers.current[item.id];
    if (!audio) return;
    Object.entries(historyAudioPlayers.current).forEach(([id, player]) => {
      if (id !== item.id && player && !player.paused) player.pause();
    });
    if (audio.src !== objectUrl) audio.src = objectUrl;
    try {
      await audio.play();
    } catch (error) {
      // Replacing a source or leaving the view can cancel playback normally.
      if (error instanceof Error && error.name === "AbortError") return;
      setHistoryError(errorMessage(error));
    }
  };

  const toggleHistoryAudio = (item: HistoryItem) => {
    const audio = historyAudioPlayers.current[item.id];
    if (audio && !audio.paused) {
      audio.pause();
    } else {
      void loadHistoryAudio(item, true);
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
      setStatusMessage("");
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

  const postSecurityAction = async (
    path: string,
    body?: Record<string, string>,
  ): Promise<AppLockStatus> => {
    const response = await apiFetch(path, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const result = await response.json() as AppLockStatus & { detail?: string };
    if (!response.ok) {
      throw new Error(result.detail ?? `Local security action failed (HTTP ${response.status}).`);
    }
    return result;
  };

  const unlockApp = async () => {
    setLockBusy(true);
    setLockError("");
    try {
      const unlocked = await postSecurityAction("/api/security/unlock", {
        password: unlockPassword,
      });
      setAppLockStatus(unlocked);
      setUnlockPassword("");
      setBootReady(false);
      await prepareWorkspace(true);
    } catch (error) {
      setUnlockPassword("");
      setLockError(errorMessage(error));
      try {
        const response = await apiFetch("/api/security/status", { cache: "no-store" });
        if (response.ok) setAppLockStatus(await response.json() as AppLockStatus);
      } catch {
        setLockError(`${errorMessage(error)} Local security status could not be refreshed.`);
      }
    } finally {
      setLockBusy(false);
    }
  };

  const setupAppLock = async () => {
    setLockBusy(true);
    setLockError("");
    try {
      const result = await postSecurityAction("/api/security/setup", {
        password: lockPassword,
        confirmation: lockConfirmation,
      });
      setAppLockStatus(result);
      setLockPassword("");
      setLockConfirmation("");
      setStatusMessage("App lock enabled. It will lock automatically the next time the app starts.");
    } catch (error) {
      setLockError(errorMessage(error));
    } finally {
      setLockBusy(false);
    }
  };

  const lockAppNow = async () => {
    setLockBusy(true);
    setLockError("");
    try {
      setAppLockStatus(await postSecurityAction("/api/security/lock"));
    } catch (error) {
      setLockError(errorMessage(error));
    } finally {
      setLockBusy(false);
    }
  };

  const disableAppLock = async () => {
    setLockBusy(true);
    setLockError("");
    try {
      const result = await postSecurityAction("/api/security/disable", {
        password: disablePassword,
      });
      setAppLockStatus(result);
      setDisablePassword("");
      setStatusMessage("App lock disabled. Local history and audio remain unchanged.");
    } catch (error) {
      setLockError(errorMessage(error));
      setDisablePassword("");
    } finally {
      setLockBusy(false);
    }
  };

  const resetAppLock = async () => {
    const warning = label(
      "Reset the app lock? This removes the PIN/password only. It does not delete history, drafts, settings, or audio.",
      "កម្មវិធីនឹងលុបតែ PIN/ពាក្យសម្ងាត់ប៉ុណ្ណោះ។ ប្រវត្តិ សេចក្តីព្រាង ការកំណត់ និងសំឡេងនឹងមិនត្រូវបានលុបទេ។",
      language,
    );
    if (!window.confirm(warning)) return;
    setLockBusy(true);
    setLockError("");
    setBootReady(false);
    try {
      const result = await postSecurityAction("/api/security/reset");
      setAppLockStatus(result);
      setUnlockPassword("");
      await prepareWorkspace(true);
      setStatusMessage(result.message ?? "App lock reset; local history was kept.");
    } catch (error) {
      setBootReady(true);
      setLockError(errorMessage(error));
    } finally {
      setLockBusy(false);
    }
  };

  const changeLanguage = async (nextLanguage: "en" | "km") => {
    setLanguage(nextLanguage);
    if (appLockStatus?.locked) return;
    try {
      const response = await apiFetch("/api/storage/settings/ui_language", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value: nextLanguage }),
      });
      if (!response.ok) throw new Error(`Language setting could not be saved (HTTP ${response.status}).`);
    } catch (error) {
      setStatusMessage(errorMessage(error));
    }
  };

  const languageToggle = (
    <div className="language-toggle" aria-label="Interface language">
      <button
        type="button"
        aria-pressed={language === "en"}
        onClick={() => void changeLanguage("en")}
      >
        EN
      </button>
      <button
        type="button"
        aria-pressed={language === "km"}
        onClick={() => void changeLanguage("km")}
      >
        ខ្មែរ
      </button>
    </div>
  );

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

  const historyGroups: { date: string; items: HistoryItem[] }[] = [];
  historyItems.forEach((item) => {
    const date = new Date(item.created_at).toLocaleDateString(language === "km" ? "km-KH" : "en-US", {
      month: "short",
      day: "numeric",
    });
    const latestGroup = historyGroups[historyGroups.length - 1];
    if (latestGroup?.date === date) {
      latestGroup.items.push(item);
    } else {
      historyGroups.push({ date, items: [item] });
    }
  });

  const lowVram = !engineStatus?.model_loaded
    && (!gpuStatus?.available || !gpuStatus?.cuda_available || !gpuStatus.safe_to_load_model);
  const gpuMessage = engineStatus?.model_device === "cuda"
    ? label(
        `VoxCPM2 is using the GPU. Keep the web TTS model stopped; ${gpuStatus?.free_mib ?? "—"} MiB is free. Other GPU apps reduce headroom and can trigger CPU fallback.`,
        `VoxCPM2 កំពុងប្រើ GPU។ សូមបិទម៉ាស៊ីន TTS របស់វេបសាយ; មាន VRAM នៅសល់ ${gpuStatus?.free_mib ?? "—"} MiB។ កម្មវិធី GPU ផ្សេងអាចបង្ខំឱ្យប្តូរទៅ CPU។`,
        language,
      )
    : engineStatus?.model_device === "cpu"
      ? label(
          "VoxCPM2 is using the much slower CPU fallback. The measured short sample took 50.88 seconds. Do not start another VoxCPM2 model in the web server.",
          "VoxCPM2 កំពុងប្រើ CPU ដែលយឺតជាងច្រើន។ សំណាកខ្លីដែលបានវាស់ចំណាយ ៥០.៨៨ វិនាទី។ សូមកុំបើក VoxCPM2 មួយទៀតនៅលើម៉ាស៊ីនមេវេបសាយ។",
          language,
        )
    : gpuStatus?.warning
      ?? "Checking NVIDIA memory. Do not load VoxCPM2 until this check completes.";

  if (!bootReady || autoLoadPending || loadingDevice !== null) {
    const loadingModel = autoLoadPending || loadingDevice !== null;
    return (
      <main className="startup-screen" aria-busy="true">
        <StartupBackdrop />
        <section className="startup-card" aria-live="polite">
          <div className="startup-topline">
            <div className="startup-heading">
              <span className="brand-mark startup-brand-mark" aria-hidden="true">
                <Image className="brand-mark-image" src="/branding/kalapak-mark-white.png" alt="" width={450} height={450} priority />
              </span>
              <div>
                <strong>KalapakTTS</strong>
                <small>{label("Offline Khmer voice studio", "កម្មវិធីបង្កើតសំឡេងខ្មែរក្នុងម៉ាស៊ីន", language)}</small>
              </div>
            </div>
            {languageToggle}
          </div>
          <div className="startup-content">
            <div className="startup-illustration"><MicrophoneIllustration /></div>
            <div className="startup-info">
              <h1>
                {label("Preparing your ", "កំពុងរៀបចំ", language)}
                <span className="text-gradient">{label("local workspace", "កន្លែងធ្វើការក្នុងម៉ាស៊ីន", language)}</span>
              </h1>
              <p>
                {label(
                  "Your offline Khmer voice studio is getting ready.",
                  "កម្មវិធីបង្កើតសំឡេងខ្មែរក្នុងម៉ាស៊ីនកំពុងរៀបចំ។",
                  language,
                )}
              </p>
              <div className="startup-progress-wrap">
                <div
                  className="startup-progress-bar"
                  role="progressbar"
                  aria-label={label("Startup progress", "ដំណើរការចាប់ផ្តើម", language)}
                >
                  <span className="startup-progress-fill" />
                </div>
              </div>
              <div className="startup-loading-status">
                <span className="loading-spinner-circle" aria-hidden="true" />
                <span className="startup-loading-text">
                  {loadingModel
                    ? label(
                        "Loading VoxCPM2 locally…",
                        "កំពុងដំណើរការ VoxCPM2 ក្នុងម៉ាស៊ីន…",
                        language,
                      )
                    : bootProgress}
                </span>
              </div>
              {statusMessage && (
                <p className="startup-message" role="status">
                  {statusMessage}
                </p>
              )}
            </div>
          </div>
        </section>
      </main>
    );
  }

  if (appLockStatus?.locked) {
    return (
      <main className="startup-screen">
        <StartupBackdrop />
        <section className="startup-card lock-card">
          <div className="startup-topline">
            <div className="startup-heading">
              <span className="brand-mark startup-brand-mark" aria-hidden="true">
                <Image className="brand-mark-image" src="/branding/kalapak-mark-white.png" alt="" width={450} height={450} priority />
              </span>
              <div>
                <strong>KalapakTTS</strong>
                <small>{label("Local app lock", "ការចាក់សោកម្មវិធីក្នុងម៉ាស៊ីន", language)}</small>
              </div>
            </div>
            {languageToggle}
          </div>
          <h1>{label("Unlock this PC profile", "ដោះសោកម្មវិធីលើកុំព្យូទ័រនេះ", language)}</h1>
          <p>{label("Enter the PIN/password you set for this app.", "បញ្ចូល PIN/ពាក្យសម្ងាត់ដែលអ្នកបានកំណត់សម្រាប់កម្មវិធីនេះ។", language)}</p>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void unlockApp();
            }}
          >
            <label className="field-label" htmlFor="unlock-password">
              {label("PIN or password", "PIN ឬពាក្យសម្ងាត់", language)}
            </label>
            <input
              id="unlock-password"
              type="password"
              autoComplete="current-password"
              value={unlockPassword}
              onChange={(event) => setUnlockPassword(event.target.value)}
              maxLength={128}
              required
            />
            <button type="submit" className="primary-button" disabled={lockBusy || unlockPassword.length === 0}>
              {lockBusy ? label("Checking…", "កំពុងត្រួតពិនិត្យ…", language) : label("Unlock", "ដោះសោ", language)}
            </button>
          </form>
          {appLockStatus.lockout_remaining_seconds > 0 && (
            <p className="inline-error" role="alert">
              {label("Try again in", "សូមព្យាយាមម្តងទៀតក្នុងរយៈពេល", language)}{" "}
              {appLockStatus.lockout_remaining_seconds} {label("seconds.", "វិនាទី។", language)}
            </p>
          )}
          {lockError && <p className="inline-error" role="alert">{lockError}</p>}
          <button type="button" className="text-button" onClick={() => void resetAppLock()} disabled={lockBusy}>
            {label("Forgot PIN/password? Reset app lock", "ភ្លេច PIN/ពាក្យសម្ងាត់? កំណត់ការចាក់សោឡើងវិញ", language)}
          </button>
          <p className="privacy-note">
            {label(
              "Reset removes only the app lock. History, drafts and audio are kept.",
              "ការកំណត់ឡើងវិញលុបតែការចាក់សោប៉ុណ្ណោះ។ ប្រវត្តិ សេចក្តីព្រាង និងសំឡេងនៅដដែល។",
              language,
            )}
          </p>
        </section>
      </main>
    );
  }

  return (
    <main className="desktop-shell" data-view={activeView} data-panel={workspacePanel} data-theme={theme}>
      <nav className="sidebar-rail" aria-label={label("Main navigation", "ម៉ឺនុយចម្បង", language)}>
        <span className="rail-brand brand-mark" aria-hidden="true">
          <Image className="brand-mark-image" src="/branding/kalapak-mark-white.png" alt="" width={450} height={450} priority />
        </span>
        <button
          type="button"
          className={`rail-button ${activeView === "generate" ? "active" : ""}`}
          onClick={() => {
            stopHistoryAudio();
            setActiveView("generate");
            setWorkspacePanel("settings");
          }}
          aria-label={label("Generate speech", "បង្កើតសំឡេង", language)}
          aria-pressed={activeView === "generate"}
        >
          <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2L15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26z"/></svg>
        </button>
        <button
          type="button"
          className={`rail-button ${activeView === "history" ? "active" : ""}`}
          onClick={() => {
            stopHistoryAudio();
            setActiveView("history");
          }}
          aria-label={label("Generation history", "ប្រវត្តិបង្កើតសំឡេង", language)}
          aria-pressed={activeView === "history"}
        >
          <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        </button>
        <span className="rail-spacer" />
      </nav>

      <header className="topbar">
        <div className="topbar-title">
          <h1>{activeView === "history" ? label("History", "ប្រវត្តិ", language) : label("Generate Speech", "បង្កើតសំឡេង", language)}</h1>
          <p>{activeView === "history" ? label("Your saved local projects", "គម្រោងដែលបានរក្សាទុកក្នុងម៉ាស៊ីន", language) : label("Create a new Khmer voice project", "បង្កើតគម្រោងសំឡេងខ្មែរថ្មី", language)}</p>
        </div>
        <div className="topbar-tools">
          {languageToggle}
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={theme === "dark" ? label("Switch to light theme", "ប្តូរទៅរូបរាងភ្លឺ", language) : label("Switch to dark theme", "ប្តូរទៅរូបរាងងងឹត", language)}
            title={theme === "dark" ? "Light theme" : "Dark theme"}
          >
            <span aria-hidden="true">{theme === "dark" ? <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg> : <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>}</span>
          </button>
          <div className="profile-status">
            <span className="status-dot" aria-hidden="true" />
            {label("This PC · Offline", "កុំព្យូទ័រនេះ · គ្មានអ៊ីនធឺណិត", language)}
          </div>
        </div>
      </header>

      <div className="workspace">
        <section className="editor-card" aria-labelledby="editor-paragraph-title">
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
          <div className="editor-header">
            <div>
              <strong id="editor-paragraph-title">{label("Untitled project", "គម្រោងគ្មានចំណងជើង", language)}</strong>
              <span>
                {text.trim()
                  ? label("Ready to generate", "រួចរាល់សម្រាប់បង្កើតសំឡេង", language)
                  : label("Type or paste your Khmer text below.", "វាយ ឬបិទភ្ជាប់អត្ថបទខ្មែររបស់អ្នកខាងក្រោម", language)}
              </span>
            </div>
            <span className="composer-mode"><span aria-hidden="true" />{label("Offline editor", "កម្មវិធីកែសម្រួលក្រៅបណ្ដាញ", language)}</span>
          </div>

          <div className="editor-composer">
            <div className="composer-toolbar">
              <span className="composer-voice">
                <span className="voice-avatar" aria-hidden="true">V</span>
                <span>{label("Vanna", "វណ្ណា", language)}</span>
              </span>
              <span className="composer-language">{label("Khmer", "ភាសាខ្មែរ", language)}</span>
              <div className="composer-actions">
                <input
                  ref={textImportInputRef}
                  type="file"
                  accept=".txt,text/plain"
                  hidden
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void importTextFile(file);
                  }}
                />
                <button
                  type="button"
                  className="composer-action-button"
                  onClick={() => textImportInputRef.current?.click()}
                  disabled={generating}
                >
                  <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <path d="M14 2v6h6M8 13h8M8 17h8" />
                  </svg>
                  <span>{label("Import", "នាំចូល", language)}</span>
                </button>
                <button
                  type="button"
                  className="composer-action-button"
                  onClick={() => setText("")}
                  disabled={!text || generating}
                >
                  <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="m7 21 9.6-9.6a2.1 2.1 0 0 0-3-3L4 18.1a2.1 2.1 0 0 0 3 2.9Z" />
                    <path d="M22 21H7M5.2 15.2 13.8 21" />
                  </svg>
                  <span>{label("Clear", "សម្អាត", language)}</span>
                </button>
              </div>
            </div>
            <label className="sr-only" htmlFor="speech-text">{label("Text to speak", "អត្ថបទសម្រាប់បង្កើតសំឡេង", language)}</label>
            <textarea
              ref={speechTextAreaRef}
              id="speech-text"
              maxLength={5000}
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder={label("Type or paste your Khmer text here…", "សូមវាយ ឬបិទភ្ជាប់អត្ថបទភាសាខ្មែរនៅទីនេះ…", language)}
            />
            <div className="editor-composer-footer">
              <span>{label("Your text stays on this PC", "អត្ថបទរបស់អ្នករក្សាទុកក្នុងកុំព្យូទ័រនេះ", language)}</span>
              <div className="character-count">{text.length.toLocaleString()} / 5,000</div>
            </div>
          </div>

          <div className="editor-actions">
            <button className="secondary-button editor-action-with-icon" type="button" onClick={() => void saveDraft()} disabled={savingDraft || storageStatus?.status !== "ready"}>
              <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z" />
                <path d="M17 21v-8H7v8M7 3v5h8" />
              </svg>
              <span>{savingDraft ? label("Saving…", "កំពុងរក្សាទុក…", language) : label("Save draft", "រក្សាទុកសេចក្តីព្រាង", language)}</span>
            </button>
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
              {generating ? label("Generating…", "កំពុងបង្កើតសំឡេង…", language) : label("Generate speech", "បង្កើតសំឡេង", language)}
            </button>
          </div>
          {(statusMessage || draftSaveState) && (
            <div className="editor-feedback">
              {statusMessage && <p className="inline-status" role="status">{statusMessage}</p>}
              {draftSaveState && (
                <p className={`draft-save-state ${draftSaveState}`} role="status">
                  {draftSaveState === "saving" ? (
                    <span className="draft-save-spinner" aria-hidden="true" />
                  ) : (
                    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="m5 12 4 4L19 6" />
                    </svg>
                  )}
                  <span>
                    {draftSaveState === "saving"
                      ? label("Saving draft…", "កំពុងរក្សាទុកសេចក្តីព្រាង…", language)
                      : draftSaveState === "saved"
                        ? label("Draft saved", "បានរក្សាទុកសេចក្តីព្រាង", language)
                        : label("Draft autosaved", "បានរក្សាទុកសេចក្តីព្រាងដោយស្វ័យប្រវត្តិ", language)}
                  </span>
                </p>
              )}
            </div>
          )}
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
        </section>

        <aside className="status-card" aria-label="Local app status">
          <div className="panel-tabs" role="tablist" aria-label={label("Workspace panels", "ផ្ទាំងកន្លែងធ្វើការ", language)}>
            <button type="button" role="tab" aria-selected={workspacePanel === "settings"} onClick={() => { stopHistoryAudio(); setWorkspacePanel("settings"); }}>
              {label("Settings", "ការកំណត់", language)}
            </button>
            <button type="button" role="tab" aria-selected={workspacePanel === "history"} onClick={() => setWorkspacePanel("history")}>
              {label("History", "ប្រវត្តិ", language)}
              <span className="history-count">{storageStatus?.generation_count ?? historyItems.length}</span>
            </button>
          </div>

          <div className="sidebar-settings">
            <section className="settings-control-card">
              <h3>{label("Voice & delivery", "សំឡេង និងការនិយាយ", language)}</h3>
              <div className="voice-setting-row">
                <span className="voice-avatar" aria-hidden="true">V</span>
                <label>
                  <span className="field-label">{label("Voice", "សំឡេង", language)}</span>
                  <select value={settings.voice} onChange={(event) => setSettings({ ...settings, voice: event.target.value })}>
                    <option value="vanna">Vanna · Khmer</option>
                  </select>
                </label>
              </div>
              <div className="settings-grid">
                <label>
                  <span className="field-label">{label("Language", "ភាសា", language)}</span>
                  <select value={settings.language} disabled>
                    <option value="km">Khmer · ភាសាខ្មែរ</option>
                  </select>
                </label>
                <label>
                  <span className="field-label">{label("Pace", "ល្បឿន", language)}</span>
                  <select value={settings.pace} onChange={(event) => setSettings({ ...settings, pace: event.target.value })}>
                    <option value="relaxed">Relaxed · 12 CPS</option>
                    <option value="natural">Natural · 14 CPS</option>
                    <option value="brisk">Brisk · 16 CPS</option>
                  </select>
                </label>
                <label>
                  <span className="field-label">{label("Quality / steps", "គុណភាព / ជំហាន", language)}</span>
                  <select value={settings.steps} onChange={(event) => setSettings({ ...settings, steps: Number(event.target.value) })}>
                    <option value={4}>Fast · 4</option>
                    <option value={8}>Balanced · 8</option>
                    <option value={12}>Quality · 12</option>
                  </select>
                </label>
                <label>
                  <span className="field-label">{label("Speed · pitch-preserving", "ល្បឿន · រក្សាកម្ពស់សំឡេង", language)}</span>
                  <select value={settings.speed} onChange={(event) => setSettings({ ...settings, speed: Number(event.target.value) })}>
                    <option value={0.85}>0.85× · Slower</option>
                    <option value={1}>1.00× · Normal</option>
                    <option value={1.15}>1.15× · Faster</option>
                  </select>
                </label>
              </div>
            </section>

            <section className="engine-control-card">
              <div className="engine-card-heading">
                <span className="eyebrow">{label("LOCAL ENGINE", "ម៉ាស៊ីនក្នុងម៉ាស៊ីន", language)}</span>
                <span className="offline-badge"><span aria-hidden="true" />{label("Offline", "ក្រៅបណ្ដាញ", language)}</span>
              </div>
              <h2>VoxCPM2</h2>
              <p>{label("This PC · Khmer speech generation", "កុំព្យូទ័រនេះ · បង្កើតសំឡេងខ្មែរ", language)}</p>
              <section className={`gpu-banner ${lowVram ? "warning" : "ready"}`} role={lowVram ? "alert" : "status"} aria-live="polite">
                <span className="banner-icon" aria-hidden="true">{lowVram ? <svg width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="none"><path d="M12 2L1 21h22L12 2zm0 15a1.5 1.5 0 110 3 1.5 1.5 0 010-3zm-1-8h2v6h-2V9z"/></svg> : <svg width="12" height="12" viewBox="0 0 24 24" fill="white" stroke="none"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg>}</span>
                <div>
                  <strong>
                    {engineStatus?.model_device === "cuda"
                      ? label("Model loaded on GPU", "ម៉ូដែលបានផ្ទុកលើ GPU", language)
                      : engineStatus?.model_device === "cpu"
                        ? label("Running on CPU fallback", "កំពុងប្រើ CPU ជំនួស", language)
                        : lowVram
                          ? label("GPU memory warning", "ការព្រមានអំពីអង្គចងចាំ GPU", language)
                          : label("GPU memory check complete", "បានពិនិត្យអង្គចងចាំ GPU", language)}
                  </strong>
                  <p>{gpuMessage}</p>
                </div>
                {gpuStatus?.available && (
                  <span className="vram-readout">
                    {label("Free", "នៅសល់", language)}: {gpuStatus.nvidia_smi_free_mib ?? gpuStatus.free_mib ?? "—"} MiB
                    {gpuStatus.model_need_mib === null ? "" : ` · ${label("Need", "ត្រូវការ", language)}: ${gpuStatus.model_need_mib} MiB`}
                  </span>
                )}
              </section>
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
                    {loadingDevice === "cuda" ? label("Loading GPU model…", "កំពុងផ្ទុកម៉ូដែល GPU…", language) : label("Load on RTX 4050", "ផ្ទុកលើ RTX 4050", language)}
                  </button>
                  {gpuStatus?.available && gpuStatus.cuda_available && !gpuStatus.safe_to_load_model && (
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => void loadEngine("cuda", true)}
                      disabled={loadingDevice !== null || !engineStatus?.model_available || storageStatus?.status !== "ready"}
                    >
                      {loadingDevice === "cuda" ? label("Trying GPU…", "កំពុងសាកល្បង GPU…", language) : label("Try GPU anyway", "សាកល្បង GPU ទោះមានការព្រមាន", language)}
                    </button>
                  )}
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => void loadEngine("cpu")}
                    disabled={loadingDevice !== null || !engineStatus?.model_available || storageStatus?.status !== "ready"}
                  >
                    {loadingDevice === "cpu" ? label("Loading CPU model…", "កំពុងផ្ទុកម៉ូដែល CPU…", language) : label("Use CPU fallback (much slower)", "ប្រើ CPU (យឺតជាងច្រើន)", language)}
                  </button>
                </div>
              )}
              <p className="engine-note">
                {engineStatus?.model_error
                  ?? engineStatus?.fallback_reason
                  ?? (engineStatus?.model_loaded
                    ? `VoxCPM2 is loaded on ${engineStatus.model_device} in BF16 mode. Stop the web TTS model before using this desktop engine.`
                    : "Load the local model once; it stays warm for this app session. CUDA is blocked below the startup VRAM threshold.")}
              </p>
            </section>

            <details className="local-status-details">
              <summary>{label("Device & storage status", "ស្ថានភាពឧបករណ៍ និងទំហំផ្ទុក", language)}</summary>
          <dl>
            <div><dt>{label("Profile", "ទម្រង់", language)}</dt><dd>{label("Automatic · no sign-in", "ស្វ័យប្រវត្តិ · មិនចាំបាច់ចូលគណនី", language)}</dd></div>
            <div><dt>{label("Storage", "ទំហំផ្ទុក", language)}</dt><dd>{storageStatus?.status === "ready" ? label("SQLite ready", "SQLite រួចរាល់", language) : storageStatus?.status === "recovery_required" ? label("Recovery required", "ត្រូវការសង្គ្រោះ", language) : label("Checking…", "កំពុងត្រួតពិនិត្យ…", language)}</dd></div>
            <div><dt>{label("Database version", "កំណែមូលដ្ឋានទិន្នន័យ", language)}</dt><dd>{storageStatus?.schema_version ?? "—"}</dd></div>
            <div><dt>{label("Profile folder", "ថតទម្រង់", language)}</dt><dd>{storageStatus?.root ?? label("Loading local path…", "កំពុងផ្ទុកទីតាំងក្នុងម៉ាស៊ីន…", language)}</dd></div>
            <div><dt>{label("Generated audio", "សំឡេងដែលបានបង្កើត", language)}</dt><dd>{formatBytes(storageStatus?.audio_storage_bytes ?? 0)}</dd></div>
            <div><dt>{label("Free disk space", "ទំហំថាសនៅសល់", language)}</dt><dd>{typeof storageStatus?.free_disk_bytes === "number" ? formatBytes(storageStatus.free_disk_bytes) : "—"}</dd></div>
            <div><dt>{label("History items", "ចំនួនប្រវត្តិ", language)}</dt><dd>{storageStatus?.generation_count ?? "—"}</dd></div>
            <div><dt>{label("GPU", "កាតក្រាហ្វិក", language)}</dt><dd>{gpuStatus?.gpu_name ?? (gpuStatus ? label("Unavailable", "មិនមាន", language) : label("Checking…", "កំពុងត្រួតពិនិត្យ…", language))}</dd></div>
            <div><dt>{label("Startup free VRAM", "VRAM ទំនេរពេលចាប់ផ្តើម", language)}</dt><dd>{startupGpuStatus?.nvidia_smi_free_mib ?? startupGpuStatus?.free_mib ?? "—"} MiB</dd></div>
            <div><dt>{label("NVIDIA driver", "កម្មវិធីបញ្ជា NVIDIA", language)}</dt><dd>{gpuStatus?.driver_version ? `${gpuStatus.driver_version} · ${label("minimum", "អប្បបរមា", language)} ${gpuStatus.minimum_driver_version}` : gpuStatus?.minimum_driver_version ?? "—"}</dd></div>
            <div><dt>{label("Engine", "ម៉ាស៊ីនបង្កើតសំឡេង", language)}</dt><dd>{engineStatus?.model_loaded ? `${engineStatus.model_device} · ${engineStatus.model_dtype ?? "loaded"}` : label("Not loaded", "មិនទាន់ផ្ទុក", language)}</dd></div>
            <div><dt>{label("App lock", "ការចាក់សោកម្មវិធី", language)}</dt><dd>{appLockStatus?.enabled ? label("Enabled", "បានបើក", language) : label("Off", "បិទ", language)}</dd></div>
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
            </details>
          <section className="app-lock-panel" aria-label={label("App lock settings", "ការកំណត់ការចាក់សោកម្មវិធី", language)}>
            <h3>{label("Optional app lock", "ការចាក់សោកម្មវិធីជាជម្រើស", language)}</h3>
            {appLockStatus?.enabled ? (
              <>
                <p>{label("The app will ask for your PIN/password the next time it starts.", "កម្មវិធីនឹងសួរ PIN/ពាក្យសម្ងាត់នៅពេលចាប់ផ្តើមលើកក្រោយ។", language)}</p>
                <button type="button" className="secondary-button" onClick={() => void lockAppNow()} disabled={lockBusy}>
                  {label("Lock now", "ចាក់សោឥឡូវនេះ", language)}
                </button>
                <form onSubmit={(event) => { event.preventDefault(); void disableAppLock(); }}>
                  <label className="field-label" htmlFor="disable-lock-password">{label("Current PIN/password", "PIN/ពាក្យសម្ងាត់បច្ចុប្បន្ន", language)}</label>
                  <input id="disable-lock-password" type="password" autoComplete="current-password" minLength={6} maxLength={128} value={disablePassword} onChange={(event) => setDisablePassword(event.target.value)} required />
                  <button type="submit" className="text-button" disabled={lockBusy}>{label("Disable app lock", "បិទការចាក់សោ", language)}</button>
                </form>
              </>
            ) : (
              <form onSubmit={(event) => { event.preventDefault(); void setupAppLock(); }}>
                <p>{label("Off by default. Use at least 6 characters. This is a privacy lock, not account security.", "បិទតាមលំនាំដើម។ ប្រើយ៉ាងតិច ៦ តួអក្សរ។ នេះជាការចាក់សោឯកជនភាព មិនមែនសុវត្ថិភាពគណនីទេ។", language)}</p>
                <label className="field-label" htmlFor="new-lock-password">{label("New PIN/password", "PIN/ពាក្យសម្ងាត់ថ្មី", language)}</label>
                <input id="new-lock-password" type="password" autoComplete="new-password" minLength={6} maxLength={128} value={lockPassword} onChange={(event) => setLockPassword(event.target.value)} required />
                <label className="field-label" htmlFor="confirm-lock-password">{label("Confirm PIN/password", "បញ្ជាក់ PIN/ពាក្យសម្ងាត់", language)}</label>
                <input id="confirm-lock-password" type="password" autoComplete="new-password" minLength={6} maxLength={128} value={lockConfirmation} onChange={(event) => setLockConfirmation(event.target.value)} required />
                <button type="submit" className="secondary-button" disabled={lockBusy}>
                  {lockBusy ? label("Saving…", "កំពុងរក្សាទុក…", language) : label("Enable app lock", "បើកការចាក់សោ", language)}
                </button>
              </form>
            )}
            {lockError && <p className="inline-error" role="alert">{lockError}</p>}
          </section>
          <p className="privacy-note">{label("Drafts and settings are stored on this PC. They are not uploaded to an account.", "សេចក្តីព្រាង និងការកំណត់ត្រូវបានរក្សាទុកនៅលើកុំព្យូទ័រនេះ។ វាមិនត្រូវបានផ្ញើទៅគណនីណាមួយទេ។", language)}</p>
          </div>
          {activeView === "generate" && workspacePanel === "history" && (
            <div className="sidebar-history" aria-label={label("Recent generation history", "ប្រវត្តិបង្កើតសំឡេងថ្មីៗ", language)}>
              <div className="sidebar-history-toolbar">
                <label className="sidebar-history-search">
                  <svg aria-hidden="true" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="7" />
                    <path d="m20 20-4-4" />
                  </svg>
                  <span className="sr-only">{label("Search history", "ស្វែងរកប្រវត្តិ", language)}</span>
                  <input
                    type="search"
                    value={historyFilters.search}
                    onChange={(event) => setHistoryFilters({ ...historyFilters, search: event.target.value })}
                    placeholder={label("Search history…", "ស្វែងរកប្រវត្តិ…", language)}
                  />
                </label>
                <button
                  type="button"
                  className="sidebar-history-refresh"
                  onClick={() => void refreshHistory(historyFilters)}
                  disabled={historyLoading}
                  aria-label={label("Refresh history", "ផ្ទុកប្រវត្តិឡើងវិញ", language)}
                  title={label("Refresh history", "ផ្ទុកប្រវត្តិឡើងវិញ", language)}
                >
                  <svg aria-hidden="true" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20 7v5h-5M4 17v-5h5" />
                    <path d="M5.6 9A7 7 0 0 1 17.5 6L20 12M4 12l2.5 6a7 7 0 0 0 11.9-3" />
                  </svg>
                </button>
              </div>
              {historyLoading && <p className="inline-status" role="status">{label("Loading history…", "កំពុងផ្ទុកប្រវត្តិ…", language)}</p>}
              {historyError && (
                <HistoryNotice
                  message={historyError}
                  language={language}
                  onDismiss={() => setHistoryError("")}
                />
              )}
              {!historyLoading && !historyError && historyItems.length === 0 && (
                <p className="history-empty">{label("No saved generations yet.", "មិនទាន់មានសំឡេងដែលបានរក្សាទុកទេ។", language)}</p>
              )}
              <div className="sidebar-history-list">
                {historyGroups.map((group) => (
                  <section className="sidebar-history-group" key={group.date}>
                    <h3>{group.date}</h3>
                    <div className="sidebar-history-group-items">
                      {group.items.map((item) => (
                        <article className="sidebar-history-item" key={item.id}>
                          <button
                            type="button"
                            className="sidebar-history-play"
                            onClick={() => toggleHistoryAudio(item)}
                            disabled={item.audio_missing || !item.audio_url || loadingAudioId === item.id}
                            aria-label={
                              loadingAudioId === item.id
                                ? label("Loading audio", "កំពុងផ្ទុកសំឡេង", language)
                                : playingHistoryId === item.id
                                  ? label("Pause audio", "ផ្អាកសំឡេង", language)
                                  : label("Play audio", "ចាក់សំឡេង", language)
                            }
                          >
                            {loadingAudioId === item.id
                              ? <span className="history-play-spinner" aria-hidden="true" />
                              : playingHistoryId === item.id
                                ? <svg aria-hidden="true" width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M7 5h4v14H7zM15 5h4v14h-4z" /></svg>
                                : <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.5v13l11-6.5z" /></svg>}
                          </button>
                          <div className="sidebar-history-item-main">
                            <button type="button" className="sidebar-history-title" onClick={() => void loadHistoryItem(item)}>
                              {item.title || label("Untitled", "គ្មានចំណងជើង", language)}
                            </button>
                            <p title={item.text}>{item.text}</p>
                            <small>
                              {item.steps} {label("steps", "ជំហាន", language)} · {item.voice_id} · {item.text.length.toLocaleString()} {label("characters", "តួអក្សរ", language)}
                            </small>
                            {item.audio_missing && <small className="sidebar-history-missing">{label("Audio missing", "បាត់ឯកសារសំឡេង", language)}</small>}
                          </div>
                          <div className="sidebar-history-actions">
                            <button
                              type="button"
                              onClick={() => void updateHistoryItem(item, "favorite")}
                              aria-label={item.is_favorite ? label("Remove favorite", "ដកចេញពីចំណូលចិត្ត", language) : label("Add favorite", "បន្ថែមទៅចំណូលចិត្ត", language)}
                              title={item.is_favorite ? label("Remove favorite", "ដកចេញពីចំណូលចិត្ត", language) : label("Add favorite", "បន្ថែមទៅចំណូលចិត្ត", language)}
                            >
                              {item.is_favorite ? "★" : "☆"}
                            </button>
                            {!item.audio_missing && (
                              <button
                                type="button"
                                onClick={() => void downloadProtectedFile(item.audio_url ?? "", `${item.id}.wav`)}
                                aria-label="Download WAV"
                                title="Download WAV"
                              >
                                <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M12 3v12m0 0 4-4m-4 4-4-4M5 17v3h14v-3" />
                                </svg>
                              </button>
                            )}
                          </div>
                          {!item.audio_missing && (
                            <audio
                              ref={(audio) => { historyAudioPlayers.current[item.id] = audio; }}
                              className="history-audio-hidden"
                              preload="none"
                              src={historyAudioUrls[item.id]}
                              onPlay={() => setPlayingHistoryId(item.id)}
                              onPause={() => setPlayingHistoryId((current) => current === item.id ? null : current)}
                              onEnded={() => setPlayingHistoryId((current) => current === item.id ? null : current)}
                              aria-hidden="true"
                            >
                              Audio playback is not supported.
                            </audio>
                          )}
                        </article>
                      ))}
                    </div>
                  </section>
                ))}
              </div>
              {historyHasMore && (
                <button
                  type="button"
                  className="sidebar-history-load-more"
                  onClick={() => void refreshHistory(historyFilters, historyItems.length, true)}
                  disabled={historyLoading}
                >
                  {label("Load more", "ផ្ទុកបន្ថែម", language)}
                </button>
              )}
              <button type="button" className="sidebar-history-view-all" onClick={() => { stopHistoryAudio(); setActiveView("history"); }}>
                {label("View full history", "មើលប្រវត្តិទាំងអស់", language)}
              </button>
            </div>
          )}
        </aside>
      </div>

      {activeView === "history" && <div className="history-stack">
      <section className="history-card" aria-labelledby="history-title">
        <div className="history-heading">
          <div>
            <h2 id="history-title">{label("Generation history", "ប្រវត្តិបង្កើតសំឡេង", language)}</h2>
            <p>{label("Search, filter, replay, rename, duplicate or remove speech saved on this PC.", "ស្វែងរក ត្រង ស្តាប់ឡើងវិញ ប្តូរឈ្មោះ ចម្លង ឬលុបសំឡេងដែលបានរក្សាទុកក្នុងម៉ាស៊ីននេះ។", language)}</p>
          </div>
          <strong>
            <span>{storageStatus?.generation_count ?? historyItems.length} {label("generations", "ការបង្កើត", language)}</span>
            <span>{formatBytes(storageStatus?.audio_storage_bytes ?? 0)} {label("stored locally", "រក្សាទុកក្នុងម៉ាស៊ីន", language)}</span>
          </strong>
        </div>

        <div className="history-toolbar">
          <label className="history-search">
            <svg aria-hidden="true" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="7" />
              <path d="m20 20-4-4" />
            </svg>
            <span className="sr-only">{label("Search history", "ស្វែងរកប្រវត្តិ", language)}</span>
            <input
              type="search"
              value={historyFilters.search}
              onChange={(event) => setHistoryFilters({ ...historyFilters, search: event.target.value })}
              placeholder={label("Search generations…", "ស្វែងរកការបង្កើត…", language)}
            />
          </label>
          <details className="history-filter-menu">
            <summary>
              <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 7h16M7 12h10m-7 5h4" />
              </svg>
              {label("Filters", "តម្រង", language)}
            </summary>
            <div className="history-filters">
              <label>
                <span className="field-label">{label("From", "ចាប់ពី", language)}</span>
                <input type="date" value={historyFilters.date_from} onChange={(event) => setHistoryFilters({ ...historyFilters, date_from: event.target.value })} />
              </label>
              <label>
                <span className="field-label">{label("To", "ដល់", language)}</span>
                <input type="date" value={historyFilters.date_to} onChange={(event) => setHistoryFilters({ ...historyFilters, date_to: event.target.value })} />
              </label>
              <label>
                <span className="field-label">{label("Voice", "សំឡេង", language)}</span>
                <select value={historyFilters.voice} onChange={(event) => setHistoryFilters({ ...historyFilters, voice: event.target.value })}>
                  <option value="">{label("All voices", "សំឡេងទាំងអស់", language)}</option>
                  <option value="vanna">Vanna</option>
                </select>
              </label>
              <label>
                <span className="field-label">{label("Pace", "ល្បឿន", language)}</span>
                <select value={historyFilters.pace} onChange={(event) => setHistoryFilters({ ...historyFilters, pace: event.target.value })}>
                  <option value="">{label("All paces", "ល្បឿនទាំងអស់", language)}</option>
                  <option value="relaxed">Relaxed</option>
                  <option value="natural">Natural</option>
                  <option value="brisk">Brisk</option>
                </select>
              </label>
              <label>
                <span className="field-label">{label("Steps", "ចំនួនជំហាន", language)}</span>
                <select value={historyFilters.steps} onChange={(event) => setHistoryFilters({ ...historyFilters, steps: event.target.value })}>
                  <option value="">{label("All steps", "ជំហានទាំងអស់", language)}</option>
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
                {label("Favorites only", "តែចំណូលចិត្ត", language)}
              </label>
            </div>
          </details>
          <button
            type="button"
            className="history-refresh-button"
            onClick={() => void refreshHistory(historyFilters)}
            disabled={historyLoading}
            aria-label={label("Refresh history", "ផ្ទុកប្រវត្តិឡើងវិញ", language)}
            title={label("Refresh history", "ផ្ទុកប្រវត្តិឡើងវិញ", language)}
          >
            <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 7v5h-5M4 17v-5h5" />
              <path d="M5.6 9A7 7 0 0 1 17.5 6L20 12M4 12l2.5 6a7 7 0 0 0 11.9-3" />
            </svg>
          </button>
        </div>

        {historyLoading && <p className="inline-status" role="status">{label("Loading local history…", "កំពុងផ្ទុកប្រវត្តិក្នុងម៉ាស៊ីន…", language)}</p>}
        {historyError && (
          <HistoryNotice
            message={historyError}
            language={language}
            onDismiss={() => setHistoryError("")}
          />
        )}
        {!historyLoading && !historyError && historyItems.length === 0 && (
          <p className="history-empty">{label("No saved generations match these filters.", "មិនមានសំឡេងដែលត្រូវនឹងលក្ខខណ្ឌស្វែងរកទេ។", language)}</p>
        )}
        <div className="history-list">
          {historyItems.map((item) => (
            <article className="history-item" key={item.id}>
              <button
                type="button"
                className="history-play-button"
                onClick={() => toggleHistoryAudio(item)}
                disabled={item.audio_missing || !item.audio_url || loadingAudioId === item.id}
                aria-label={
                  loadingAudioId === item.id
                    ? label("Loading audio", "កំពុងផ្ទុកសំឡេង", language)
                    : playingHistoryId === item.id
                      ? label("Pause audio", "ផ្អាកសំឡេង", language)
                      : label("Play audio", "ចាក់សំឡេង", language)
                }
                title={
                  loadingAudioId === item.id
                    ? label("Loading audio…", "កំពុងផ្ទុកសំឡេង…", language)
                    : playingHistoryId === item.id
                      ? label("Pause audio", "ផ្អាកសំឡេង", language)
                      : label("Play audio", "ចាក់សំឡេង", language)
                }
              >
                {loadingAudioId === item.id
                  ? <span className="history-play-spinner" aria-hidden="true" />
                  : playingHistoryId === item.id
                    ? <svg aria-hidden="true" width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M7 5h4v14H7zM15 5h4v14h-4z" /></svg>
                    : <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.5v13l11-6.5z" /></svg>}
              </button>
              <div className="history-item-main">
                <div className="history-item-heading">
                  <h3>{item.title || label("Untitled", "គ្មានចំណងជើង", language)}</h3>
                  <span className={`history-status-badge ${item.audio_missing ? "missing" : ""}`}>
                    {item.audio_missing ? label("Audio missing", "បាត់ឯកសារសំឡេង", language) : label("Completed", "បានបញ្ចប់", language)}
                  </span>
                </div>
                <p className="history-text">{item.text}</p>
                <small className="history-item-meta">
                  <span>{item.voice_id}</span>
                  <span>{item.audio_duration ? `${item.audio_duration.toFixed(1)} s` : "—"}</span>
                  <span>{item.text.length.toLocaleString()} {label("characters", "តួអក្សរ", language)}</span>
                  <time dateTime={item.created_at}>{new Date(item.created_at).toLocaleDateString(language === "km" ? "km-KH" : undefined)}</time>
                </small>
                {item.audio_missing ? (
                  <p className="missing-audio" role="alert">{label("Audio file missing. Load the text and settings to regenerate.", "ឯកសារសំឡេងបាត់។ សូមផ្ទុកអត្ថបទ និងការកំណត់ដើម្បីបង្កើតឡើងវិញ។", language)}</p>
                ) : (
                  <audio
                    ref={(audio) => { historyAudioPlayers.current[item.id] = audio; }}
                    className="history-audio-hidden"
                    preload="none"
                    src={historyAudioUrls[item.id]}
                    onPlay={() => setPlayingHistoryId(item.id)}
                    onPause={() => setPlayingHistoryId((current) => current === item.id ? null : current)}
                    onEnded={() => setPlayingHistoryId((current) => current === item.id ? null : current)}
                    aria-hidden="true"
                  >
                    Audio playback is not supported.
                  </audio>
                )}
              </div>
              <div className="history-actions">
                {!item.audio_missing && (
                  <button
                    type="button"
                    className="history-icon-button history-download-button"
                    onClick={() => void downloadProtectedFile(item.audio_url ?? "", `${item.id}.wav`)}
                    aria-label="Download WAV"
                    title="Download WAV"
                  >
                    <svg aria-hidden="true" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 3v12m0 0 4-4m-4 4-4-4M5 17v3h14v-3" />
                    </svg>
                  </button>
                )}
                <details className="history-more-actions">
                  <summary aria-label={label("More history actions", "សកម្មភាពប្រវត្តិបន្ថែម", language)} title={label("More actions", "សកម្មភាពបន្ថែម", language)}>
                    <svg aria-hidden="true" width="17" height="17" viewBox="0 0 24 24" fill="currentColor">
                      <circle cx="12" cy="5" r="1.7" />
                      <circle cx="12" cy="12" r="1.7" />
                      <circle cx="12" cy="19" r="1.7" />
                    </svg>
                  </summary>
                  <div className="history-more-menu">
                    <button type="button" onClick={() => void loadHistoryItem(item)}>
                      {item.audio_missing ? label("Regenerate missing audio", "បង្កើតសំឡេងដែលបាត់ឡើងវិញ", language) : label("Open / regenerate", "បើក / បង្កើតឡើងវិញ", language)}
                    </button>
                    {!item.audio_missing && (
                      <>
                        <button type="button" onClick={() => void downloadProtectedFile(item.srt_url, `${item.id}.srt`)}>Download SRT</button>
                        <button type="button" onClick={() => void downloadProtectedFile(item.vtt_url, `${item.id}.vtt`)}>Download VTT</button>
                      </>
                    )}
                    <button type="button" onClick={() => void updateHistoryItem(item, "favorite")}>
                      {item.is_favorite ? "★ Favorite" : "☆ Favorite"}
                    </button>
                    <button type="button" onClick={() => void updateHistoryItem(item, "rename")}>{label("Rename", "ប្តូរឈ្មោះ", language)}</button>
                    <button type="button" onClick={() => void updateHistoryItem(item, "duplicate")}>{label("Duplicate", "ចម្លង", language)}</button>
                    <button type="button" className="danger" onClick={() => void updateHistoryItem(item, "delete")}>{label("Delete", "លុប", language)}</button>
                  </div>
                </details>
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
            {label("Load more history", "ផ្ទុកប្រវត្តិបន្ថែម", language)}
          </button>
        )}
      </section>

      <section className="storage-tools" aria-label="Local media storage tools">
        <div>
          <span className="eyebrow">LOCAL STORAGE</span>
          <h2>{label("Storage, backup and cleanup", "ទំហំផ្ទុក បម្រុងទុក និងសម្អាត", language)}</h2>
          <p>{label("Backups include local history, drafts, settings, voice files and generated audio with SHA-256 verification.", "ឯកសារបម្រុងទុករួមមានប្រវត្តិ សេចក្តីព្រាង ការកំណត់ សំឡេងយោង និងសំឡេងដែលបានបង្កើត ជាមួយការផ្ទៀងផ្ទាត់ SHA-256។", language)}</p>
        </div>
        <label className="media-folder-control">
          <span className="field-label">{label("Media folder path", "ទីតាំងថតឯកសារសំឡេង", language)}</span>
          <input value={mediaFolderPath} onChange={(event) => setMediaFolderPath(event.target.value)} />
        </label>
        <button type="button" className="secondary-button" onClick={() => void moveMediaFolder()} disabled={movingMedia || storageStatus?.status !== "ready"}>
          {movingMedia ? label("Moving audio…", "កំពុងផ្លាស់ទីសំឡេង…", language) : label("Move media folder", "ផ្លាស់ទីថតសំឡេង", language)}
        </button>
        <label>
          <span className="field-label">{label("Clean history before", "សម្អាតប្រវត្តិមុនថ្ងៃ", language)}</span>
          <input type="date" value={cleanupBefore} onChange={(event) => setCleanupBefore(event.target.value)} />
        </label>
        <button type="button" className="secondary-button cleanup-button" onClick={() => void cleanOldHistory()} disabled={cleaningHistory || storageStatus?.status !== "ready"}>
          {cleaningHistory ? label("Cleaning…", "កំពុងសម្អាត…", language) : label("Clean old history", "សម្អាតប្រវត្តិចាស់", language)}
        </button>
        <div className="backup-actions">
          <button type="button" className="secondary-button" onClick={() => void exportBackup()} disabled={backupBusy || storageStatus?.status !== "ready"}>
            {backupBusy ? label("Preparing backup…", "កំពុងរៀបចំបម្រុងទុក…", language) : label("Export backup ZIP", "នាំចេញ ZIP បម្រុងទុក", language)}
          </button>
          <button type="button" className="secondary-button" onClick={() => fileInputRef.current?.click()} disabled={backupBusy}>
            {backupBusy ? label("Restoring backup…", "កំពុងស្តារបម្រុងទុក…", language) : label("Import backup ZIP", "នាំចូល ZIP បម្រុងទុក", language)}
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
      </div>}
    </main>
  );
}
