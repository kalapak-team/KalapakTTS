"use client";

import { useCallback, useEffect, useState } from "react";

type GpuStatus = {
  available: boolean;
  gpu_name: string | null;
  free_mib: number | null;
  total_mib: number | null;
  minimum_free_mib: number;
  safe_to_load_model: boolean;
  warning: string;
  checked_at: string;
};

type StorageStatus = {
  status: "ready" | "recovery_required";
  root?: string;
  schema_version?: number;
  backup_available: boolean;
  latest_backup?: string | null;
  message?: string;
};

type DraftSettings = {
  voice: string;
  pace: string;
  steps: number;
};

const initialSettings: DraftSettings = {
  voice: "vanna",
  pace: "natural",
  steps: 4,
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "An unexpected local service error occurred.";
}

export default function Home() {
  const [text, setText] = useState("");
  const [settings, setSettings] = useState<DraftSettings>(initialSettings);
  const [startupGpuStatus, setStartupGpuStatus] = useState<GpuStatus | null>(null);
  const [gpuStatus, setGpuStatus] = useState<GpuStatus | null>(null);
  const [storageStatus, setStorageStatus] = useState<StorageStatus | null>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [savingDraft, setSavingDraft] = useState(false);
  const [restoringBackup, setRestoringBackup] = useState(false);

  const refreshGpuStatus = useCallback(async () => {
    const response = await fetch("/api/system/gpu", { cache: "no-store" });
    if (!response.ok) throw new Error(`GPU status check failed (HTTP ${response.status}).`);
    const payload = await response.json() as { startup: GpuStatus; current: GpuStatus };
    setStartupGpuStatus((previous) => previous ?? payload.startup);
    setGpuStatus(payload.current);
  }, []);

  const refreshStorageStatus = useCallback(async () => {
    const response = await fetch("/api/storage/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Local storage check failed (HTTP ${response.status}).`);
    const status = await response.json() as StorageStatus;
    setStorageStatus(status);
    if (status.status === "ready") {
      const draftResponse = await fetch("/api/storage/drafts/current", { cache: "no-store" });
      if (draftResponse.ok) {
        const draft = await draftResponse.json() as { text: string; settings: DraftSettings };
        setText(draft.text);
        setSettings({ ...initialSettings, ...draft.settings });
      } else if (draftResponse.status !== 404) {
        throw new Error(`Local draft could not be loaded (HTTP ${draftResponse.status}).`);
      }
    }
  }, []);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        await refreshGpuStatus();
        await refreshStorageStatus();
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
  }, [refreshGpuStatus, refreshStorageStatus]);

  const saveDraft = async () => {
    setSavingDraft(true);
    setStatusMessage("");
    try {
      const response = await fetch("/api/storage/drafts/current", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, settings }),
      });
      if (!response.ok) throw new Error(`Draft could not be saved (HTTP ${response.status}).`);
      setStatusMessage("Draft saved on this PC.");
    } catch (error) {
      setStatusMessage(errorMessage(error));
    } finally {
      setSavingDraft(false);
    }
  };

  const restoreBackup = async () => {
    const backupName = storageStatus?.latest_backup ?? "the latest available backup";
    if (!window.confirm(`Restore ${backupName}? The current database will be replaced by this backup.`)) return;

    setRestoringBackup(true);
    setStatusMessage("");
    try {
      const response = await fetch("/api/storage/restore", { method: "POST" });
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

  const lowVram = !gpuStatus?.available || !gpuStatus.safe_to_load_model;

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
          <strong>{lowVram ? "Desktop speech generation is blocked by the VRAM guard" : "GPU memory check complete"}</strong>
          <p>
            {gpuStatus
              ? gpuStatus.warning
              : "Checking NVIDIA memory. Do not start desktop speech generation until this check completes."}
          </p>
        </div>
        {gpuStatus?.available && (
          <span className="vram-readout">
            Start: {startupGpuStatus?.free_mib ?? "—"} MiB · Now: {gpuStatus.free_mib} / {gpuStatus.total_mib} MiB
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
            <span className="offline-badge"><span aria-hidden="true" />Offline app</span>
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
          </div>

          <div className="editor-actions">
            <button className="primary-button" type="button" disabled>
              Generate speech
            </button>
            <button className="secondary-button" type="button" onClick={() => void saveDraft()} disabled={savingDraft || storageStatus?.status !== "ready"}>
              {savingDraft ? "Saving…" : "Save local draft"}
            </button>
          </div>
          <p className="engine-note">The local VoxCPM2 engine is not connected yet. Speech generation stays disabled until its VRAM guard is active.</p>
          {statusMessage && <p className="inline-status" role="status">{statusMessage}</p>}
        </section>

        <aside className="status-card" aria-label="Local app status">
          <span className="eyebrow">THIS PC</span>
          <h2>Local app status</h2>
          <dl>
            <div><dt>Profile</dt><dd>Automatic · no sign-in</dd></div>
            <div><dt>Storage</dt><dd>{storageStatus?.status === "ready" ? "SQLite ready" : storageStatus?.status === "recovery_required" ? "Recovery required" : "Checking…"}</dd></div>
            <div><dt>Database version</dt><dd>{storageStatus?.schema_version ?? "—"}</dd></div>
            <div><dt>Media folder</dt><dd>{storageStatus?.root ?? "Loading local path…"}</dd></div>
            <div><dt>GPU</dt><dd>{gpuStatus?.gpu_name ?? (gpuStatus ? "Unavailable" : "Checking…")}</dd></div>
          </dl>
          <p className="privacy-note">Drafts and settings are stored on this PC. They are not uploaded to an account.</p>
        </aside>
      </div>
    </main>
  );
}
