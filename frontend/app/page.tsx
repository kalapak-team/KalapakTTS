"use client";

import { FormEvent, KeyboardEvent as ReactKeyboardEvent, useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import Image from "next/image";
import { createPortal } from "react-dom";
import {
  deleteCachedModel,
  downloadAndVerifyModel,
  getLocalAdapterStatus,
  getModelStorageInfo,
  isModelCached,
  requestPersistentStorage,
  type LocalAdapterStatus,
  type LocalModelStatus,
} from "./local-model";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const MAX_TEXT_LENGTH = 5000;

type User = {
  id: string;
  username: string;
  email: string;
  full_name?: string | null;
  role: string;
};

type Voice = {
  id: string;
  name: string;
  description?: string | null;
  control_instruction?: string | null;
  has_reference_audio: boolean;
};

type HistoryItem = {
  id: string;
  title?: string | null;
  text_input: string;
  control_instruction?: string | null;
  voice_id?: string | null;
  audio_duration?: number | null;
  cfg_value?: number;
  inference_timesteps?: number;
  speed: number;
  pace_mode: "auto" | "custom";
  target_cps?: number | null;
  pronunciation_dictionary: boolean;
  noise_reduction?: boolean;
  style?: string | null;
  status?: string;
  created_at: string;
  audio_url: string;
};

type HistoryPage = {
  items: HistoryItem[];
  total?: number;
};

type PaceMode = "auto" | "custom";
type PaceProfile = "relaxed" | "natural" | "brisk" | "custom";
type UserPreferences = {
  speed: number;
  style: string;
  paceMode: PaceMode;
  paceProfile: PaceProfile;
  targetCps: number | null;
  pronunciationDictionary: boolean;
  noiseReduction: boolean;
};

type GenerationOverrides = {
  voiceId?: string | null;
  instruction?: string | null;
  guidance?: number;
  steps?: number;
  speed?: number;
  paceMode?: PaceMode;
  targetCps?: number | null;
  pronunciationDictionary?: boolean;
  noiseReduction?: boolean;
  speakingStyle?: string | null;
};

type GenerationJob = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  current_chunk: number;
  total_chunks: number;
  message: string;
  queued_position?: number | null;
  pace_retry_count?: number;
  generation?: { id: string; audio_url: string; duration?: number; created_at: string } | null;
  metrics?: {
    speech_only_chars_per_second?: number | null;
    total_chars_per_second?: number | null;
    pause_count_250ms?: number;
    pause_seconds_250ms?: number;
    audio_duration_seconds?: number;
    pace_retry_count?: number;
  };
  error?: string | null;
};

type ModelStatus = {
  model_loaded: boolean;
  model_loading: boolean;
  model_device?: string | null;
  selected_device?: string | null;
  configured_device?: string;
  force_cuda?: boolean;
  min_free_vram_gib?: number;
  torch_version?: string | null;
  cuda_version?: string | null;
  cuda_available?: boolean;
  bf16_supported?: boolean;
  gpu_name?: string | null;
  model_dtype?: string | null;
  vae_device?: string | null;
  fallback_reason?: string | null;
  vram_used_bytes?: number;
  vram_total_bytes?: number;
  vram_allocated_bytes?: number;
  vram_reserved_bytes?: number;
  model_load_seconds?: number | null;
  last_inference_seconds?: number | null;
  peak_ram_bytes?: number | null;
  peak_vram_bytes?: number;
  downloaded_bytes: number;
  total_bytes: number;
  download_percent: number;
};

type Session = { token: string; user: User };

const styleSample = "សួស្តី! សូមស្វាគមន៍មកកាន់កម្មវិធីរបស់យើង។";
type AppTheme = "light" | "dark";

const themeSubscribers = new Set<() => void>();

function subscribeToTheme(onChange: () => void) {
  const onStorageChange = () => onChange();
  themeSubscribers.add(onChange);
  window.addEventListener("storage", onStorageChange);
  return () => {
    themeSubscribers.delete(onChange);
    window.removeEventListener("storage", onStorageChange);
  };
}

function getSavedTheme(): AppTheme {
  return localStorage.getItem("kalapaktts-theme") === "dark" ? "dark" : "light";
}

function getServerTheme(): AppTheme {
  return "light";
}

function saveTheme(theme: AppTheme) {
  localStorage.setItem("kalapaktts-theme", theme);
  themeSubscribers.forEach((subscriber) => subscriber());
}

const speakingStyles = [
  {
    id: "natural",
    label: "Natural · ធម្មជាតិ",
    speed: 1.0,
    instruction: "Speak Khmer in a calm, warm, natural voice at a steady conversational pace. Articulate words clearly, connect phrases smoothly, and use gentle natural intonation. Avoid rushing, overly long pauses, or drawn-out vowels. និយាយភាសាខ្មែរដោយសំឡេងស្ងប់ កក់ក្តៅ និងធម្មជាតិ ក្នុងល្បឿនសន្ទនាថេរ។ បញ្ចេញពាក្យឱ្យច្បាស់ តភ្ជាប់ឃ្លាឱ្យរលូន និងប្រើសូរឡើងចុះតាមធម្មជាតិ។ កុំប្រញាប់និយាយ កុំឈប់យូរពេក និងកុំពន្លូតស្រៈខ្លាំងពេក។",
  },
  {
    id: "news_anchor",
    label: "News anchor · អ្នកអានព័ត៌មាន",
    speed: 0.95,
    instruction: "Speak like a clear, formal, steady, authoritative news presenter. និយាយបែបអ្នកអានព័ត៌មាន ឱ្យច្បាស់ ផ្លូវការ ស្ងប់ និងមានទំនុកចិត្ត។",
  },
  {
    id: "vlogger",
    label: "Vlogger / Creator · អ្នកបង្កើតមាតិកា",
    speed: 1.1,
    instruction: "Use an energetic, friendly, casual, expressive content-creator tone. និយាយបែបអ្នកបង្កើតមាតិកា ឱ្យស្វាហាប់ រួសរាយ និងមានភាពរស់រវើក។",
  },
  {
    id: "storyteller",
    label: "Storyteller · អ្នកនិទានរឿង",
    speed: 0.9,
    instruction: "Use a warm, gentle storytelling voice with expressive phrasing and pauses. និយាយបែបអ្នកនិទានរឿង ឱ្យកក់ក្តៅ ទន់ភ្លន់ និងមានចន្លោះឃ្លាសមរម្យ។",
  },
  {
    id: "comedy",
    label: "Comedy · កំប្លែង",
    speed: 1.1,
    instruction: "Use a playful, lively tone with light comedic emphasis. និយាយបែបកំប្លែង ឱ្យរស់រវើក លេងសើច និងសង្កត់សំឡេងបែបកំប្លែងស្រាលៗ។",
  },
  {
    id: "teacher",
    label: "Teacher · គ្រូបង្រៀន",
    speed: 0.9,
    instruction: "Speak calmly, clearly, and patiently like a helpful teacher explaining a lesson. និយាយបែបគ្រូបង្រៀន ឱ្យស្ងប់ ច្បាស់ និងអត់ធ្មត់។",
  },
  {
    id: "advertisement",
    label: "Advertisement · ផ្សព្វផ្សាយ",
    speed: 1.05,
    instruction: "Use an upbeat, persuasive, confident advertising voice. និយាយបែបផ្សព្វផ្សាយ ឱ្យរីករាយ មានភាពទាក់ទាញ និងជឿជាក់។",
  },
  {
    id: "meditation",
    label: "Meditation · សមាធិ",
    speed: 0.8,
    instruction: "Speak softly and slowly with a soothing, calm presence. និយាយយឺតៗ ទន់ភ្លន់ និងស្ងប់ស្ងាត់បែបណែនាំសមាធិ។",
  },
  {
    id: "custom",
    label: "Custom · ផ្ទាល់ខ្លួន",
    instruction: "",
  },
] as const;

const speedPresets = [
  { label: "Very slow · យឺតខ្លាំង", value: 0.75 },
  { label: "Slow · យឺត", value: 0.9 },
  { label: "Normal · ធម្មតា", value: 1.0 },
  { label: "Fast · លឿន", value: 1.15 },
];

const paceProfileTargets: Record<Exclude<PaceProfile, "custom">, number> = {
  relaxed: 12,
  natural: 14,
  brisk: 16,
};

function instructionForStyle(voice: Voice | undefined, styleId: string) {
  const voiceInstruction = voice?.control_instruction?.trim() ?? "";
  const style = speakingStyles.find((item) => item.id === styleId);
  return [voiceInstruction, style?.instruction].filter(Boolean).join(" ");
}

function readUserPreferences(userId: string): UserPreferences {
  const storageKey = `kalapaktts-preferences-${userId}`;
  const defaults: UserPreferences = {
    speed: 1.0,
    style: "natural",
    paceMode: "custom",
    paceProfile: "natural",
    targetCps: 14,
    pronunciationDictionary: true,
    noiseReduction: false,
  };
  try {
    const stored = localStorage.getItem(storageKey);
    if (!stored) return defaults;
    const saved: Partial<Record<keyof UserPreferences, unknown>> & {
      pacePreferenceVersion?: unknown;
      noisePreferenceVersion?: unknown;
    } = JSON.parse(stored);
    const speed = typeof saved.speed === "number" && Number.isFinite(saved.speed) && saved.speed >= 0.7 && saved.speed <= 1.5
      ? saved.speed
      : defaults.speed;
    const style = typeof saved.style === "string" && speakingStyles.some((item) => item.id === saved.style)
      ? saved.style
      : defaults.style;
    const paceMode = saved.paceMode === "auto" || saved.paceMode === "custom"
      ? saved.paceMode
      : defaults.paceMode;
    const targetCps = typeof saved.targetCps === "number" && Number.isFinite(saved.targetCps) && saved.targetCps >= 5 && saved.targetCps <= 40
        ? saved.targetCps
        : defaults.targetCps;
    const paceProfile = saved.paceProfile === "relaxed"
      || saved.paceProfile === "natural"
      || saved.paceProfile === "brisk"
      || saved.paceProfile === "custom"
      ? saved.paceProfile
      : targetCps === 12
        ? "relaxed"
        : targetCps === 14
          ? "natural"
          : targetCps === 16
            ? "brisk"
            : "custom";
    const pacePreferenceVersion = typeof saved.pacePreferenceVersion === "number"
      ? saved.pacePreferenceVersion
      : 0;
    const shouldMigrateFastPace = pacePreferenceVersion < 2
      && targetCps !== null
      && targetCps > paceProfileTargets.natural;
    const migratedTargetCps = shouldMigrateFastPace
      ? paceProfileTargets.natural
      : paceProfile === "relaxed" && targetCps === 12
        ? 12
        : paceProfile === "natural" && targetCps === 14
          ? 14
          : paceProfile === "brisk" && targetCps === 16
            ? 16
            : targetCps;
    const migratedPaceProfile = shouldMigrateFastPace ? "natural" : paceProfile;
    const pronunciationDictionary = typeof saved.pronunciationDictionary === "boolean"
      ? saved.pronunciationDictionary
      : defaults.pronunciationDictionary;
    const noiseReduction = saved.noisePreferenceVersion === 2
      && typeof saved.noiseReduction === "boolean"
      ? saved.noiseReduction
      : defaults.noiseReduction;
    return {
      speed,
      style,
      paceMode,
      paceProfile: migratedPaceProfile,
      targetCps: migratedTargetCps,
      pronunciationDictionary,
      noiseReduction,
    };
  } catch {
    localStorage.removeItem(storageKey);
    return defaults;
  }
}

const iconPaths: Record<string, string> = {
  brand: "M12 2.5 21 7.7v8.6l-9 5.2-9-5.2V7.7L12 2.5Zm0 4.1-5.4 3.1v5.6l5.4 3.1 5.4-3.1V9.7L12 6.6Zm0 2.7 2.7 1.6v3.2L12 15.7l-2.7-1.6v-3.2L12 9.3Z",
  sound: "M3 10v4h4l5 4V6l-5 4H3Zm12.5-2.5a6 6 0 0 1 0 9m2.5-11.5a10 10 0 0 1 0 14",
  wave: "M2 12h2l2-7 4 14 4-14 3 14 2-7h3",
  document: "M6 3h8l4 4v14H6zM14 3v5h5M9 12h6M9 16h6",
  eraser: "m7 21-4.3-4.3a2.4 2.4 0 0 1 0-3.4L13.3 2.7a2.4 2.4 0 0 1 3.4 0l4.6 4.6a2.4 2.4 0 0 1 0 3.4L12 22H7m-5-5h12",
  music: "M9 18V5l12-2v13M9 18a3 3 0 1 1-3-3c1.7 0 3 .8 3 3Zm12-2a3 3 0 1 1-3-3c1.7 0 3 .8 3 3Z",
  mic: "M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Zm-7 9v1a7 7 0 0 0 14 0v-1m-7 8v3m-4 0h8",
  clock: "M12 8v4l3 2m6-2a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  chart: "M4 19V5m0 14h17M8 15v-4m4 4V7m4 8v-6",
  calendar: "M8 2v4m8-4v4M3 10h18M5 4h14a2 2 0 0 1 2 2v14H3V6a2 2 0 0 1 2-2Z",
  refresh: "M20 7v5h-5M4 17v-5h5m-4.1-3a8 8 0 0 1 13.2-2L20 12M4 12l1.9 5a8 8 0 0 0 13.2-2",
  copy: "M8 8V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2h-3M5 8h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2Z",
  settings: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm0-6v2m0 16v2M4.2 4.2l1.4 1.4m12.8 12.8 1.4 1.4M2 12h2m16 0h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4",
  bell: "M18 9a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9m-8 12h4",
  globe: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Zm-8-10h16M12 2a15 15 0 0 1 0 20m0-20a15 15 0 0 0 0 20",
  search: "m21 21-4.3-4.3m2.3-5.2a7.5 7.5 0 1 1-15 0 7.5 7.5 0 0 1 15 0Z",
  check: "m5 12 4 4L19 6",
  book: "M4 4.5A2.5 2.5 0 0 1 6.5 2H20v17H6.5A2.5 2.5 0 0 0 4 21.5zm0 0v17",
  chat: "M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8A8.5 8.5 0 0 1 8.7 4a8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.4Z",
  language: "m5 8h8M9 5v3c0 4-2 7-5 9m1-5c1 2 3 4 5 5m2-4 3-8 3 8m-5-2h4m-7 8h10",
  sparkles: "m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Zm7 12 .9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15Z",
  moon: "M20.9 13A9 9 0 0 1 11 3.1 9 9 0 1 0 20.9 13Z",
  sun: "M12 3v2m0 14v2M5.64 5.64l1.42 1.42m9.88 9.88 1.42 1.42M3 12h2m14 0h2M5.64 18.36l1.42-1.42m9.88-9.88 1.42-1.42M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z",
  chevron: "m7 10 5 5 5-5",
  close: "m18 6-12 12M6 6l12 12",
  edit: "m16 4 4 4M4 20l4-.8L19 8a2.1 2.1 0 0 0-3-3L5 16l-1 4Z",
  logout: "M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4m7 14 5-5-5-5m5 5H9",
  plus: "M12 5v14m-7-7h14",
  download: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4m4-5 5 5 5-5m-5 5V3",
  play: "m8 5 12 7-12 7V5Z",
  user: "M20 21a8 8 0 0 0-16 0m8-10a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z",
};

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  return (
    <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d={iconPaths[name] ?? iconPaths.sparkles} />
    </svg>
  );
}

type DropdownOption = {
  value: string;
  label: string;
  disabled?: boolean;
};

function ModernDropdown({
  id,
  label,
  value,
  options,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  options: DropdownOption[];
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<{ top: number; left: number; width: number; maxHeight: number } | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const selected = options.find((option) => option.value === value);
  const firstEnabledIndex = options.findIndex((option) => !option.disabled);

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;
    if (!trigger) return;
    const rect = trigger.getBoundingClientRect();
    const zoom = Number.parseFloat(window.getComputedStyle(document.documentElement).zoom);
    const scale = Number.isFinite(zoom) && zoom > 0 ? zoom : 1;
    const viewportWidth = window.innerWidth / scale;
    const viewportHeight = window.innerHeight / scale;
    const triggerLeft = rect.left / scale;
    const triggerTop = rect.top / scale;
    const triggerBottom = rect.bottom / scale;
    const triggerWidth = rect.width / scale;
    const maxHeight = Math.min(240, viewportHeight - 24);
    const desiredHeight = Math.min(options.length * 40 + 12, maxHeight);
    const roomBelow = viewportHeight - triggerBottom - 8;
    const opensAbove = roomBelow < desiredHeight && triggerTop > desiredHeight + 8;
    const width = Math.min(triggerWidth, viewportWidth - 24);
    const left = Math.max(12, Math.min(triggerLeft, viewportWidth - width - 12));
    const top = opensAbove
      ? Math.max(12, triggerTop - desiredHeight - 6)
      : Math.min(triggerBottom + 6, viewportHeight - desiredHeight - 12);
    setPosition({ top, left, width, maxHeight });
  }, [options.length]);

  useEffect(() => {
    if (!open) return;
    const frame = window.requestAnimationFrame(() => optionRefs.current[firstEnabledIndex]?.focus({ preventScroll: true }));
    const closeOnOutsideClick = (event: PointerEvent) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target) && !menuRef.current?.contains(event.target)) {
        setOpen(false);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    window.addEventListener("resize", updatePosition);
    document.addEventListener("scroll", updatePosition, true);
    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", updatePosition);
      document.removeEventListener("scroll", updatePosition, true);
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open, firstEnabledIndex, updatePosition]);

  const moveFocus = (event: ReactKeyboardEvent<HTMLButtonElement>, currentIndex: number) => {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const enabled = options.reduce<number[]>((indexes, option, index) => {
      if (!option.disabled) indexes.push(index);
      return indexes;
    }, []);
    if (!enabled.length) return;
    let targetIndex: number;
    if (event.key === "Home") targetIndex = enabled[0];
    else if (event.key === "End") targetIndex = enabled[enabled.length - 1];
    else {
      const currentPosition = enabled.indexOf(currentIndex);
      const direction = event.key === "ArrowDown" ? 1 : -1;
      targetIndex = enabled[(currentPosition + direction + enabled.length) % enabled.length];
    }
    optionRefs.current[targetIndex]?.focus();
  };

  return (
    <div className="modern-select" ref={rootRef}>
      <button
        id={id}
        ref={triggerRef}
        className="modern-select-trigger"
        type="button"
        role="combobox"
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={`${id}-menu`}
        data-value={value}
        onClick={() => {
          if (open) setOpen(false);
          else {
            updatePosition();
            setOpen(true);
          }
        }}
      >
        <span className="modern-select-value">{selected?.label ?? "Select an option"}</span>
        <span className={`modern-select-chevron ${open ? "open" : ""}`}><Icon name="chevron" size={16} /></span>
      </button>
      {open && position && typeof document !== "undefined" && createPortal(
        <div
          ref={menuRef}
          id={`${id}-menu`}
          className="modern-select-menu"
          role="listbox"
          aria-label={label}
          style={{ top: position.top, left: position.left, width: position.width, maxHeight: position.maxHeight }}
        >
          {options.map((option, index) => (
            <button
              key={option.value}
              ref={(element) => { optionRefs.current[index] = element; }}
              className={`modern-select-option ${option.value === value ? "selected" : ""}`}
              type="button"
              role="option"
              aria-selected={option.value === value}
              data-value={option.value}
              disabled={option.disabled}
              onKeyDown={(event) => moveFocus(event, index)}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
                triggerRef.current?.focus();
              }}
            >
              <span>{option.label}</span>
            </button>
          ))}
        </div>,
        document.body,
      )}
    </div>
  );
}

async function responseError(response: Response) {
  const body = await response.json().catch(() => null);
  return typeof body?.detail === "string" ? body.detail : `Request failed (${response.status})`;
}

function formatHistoryDuration(seconds?: number | null) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const totalSeconds = Math.round(seconds);
  return `${Math.floor(totalSeconds / 60)}m ${totalSeconds % 60}s`;
}

export default function Home() {
  const [session, setSession] = useState<Session | null>(null);
  const [text, setText] = useState("");
  const [voices, setVoices] = useState<Voice[]>([]);
  const [voiceId, setVoiceId] = useState("");
  const [voiceMenuOpen, setVoiceMenuOpen] = useState(false);
  const [voiceSearch, setVoiceSearch] = useState("");
  const [instruction, setInstruction] = useState("");
  const [guidance, setGuidance] = useState(2);
  const [steps, setSteps] = useState(10);
  const [qualityPreset, setQualityPreset] = useState("high_quality");
  const [speed, setSpeed] = useState(1.0);
  const [paceMode, setPaceMode] = useState<PaceMode>("custom");
  const [paceProfile, setPaceProfile] = useState<PaceProfile>("natural");
  const [targetCps, setTargetCps] = useState<number | null>(14);
  const [lastMeasuredCps, setLastMeasuredCps] = useState<number | null>(null);
  const [lastGenerationMetrics, setLastGenerationMetrics] = useState<GenerationJob["metrics"] | null>(null);
  const [pronunciationDictionary, setPronunciationDictionary] = useState(true);
  const [noiseReduction, setNoiseReduction] = useState(false);
  const [speakingStyle, setSpeakingStyle] = useState("natural");
  const [activeTab, setActiveTab] = useState<"settings" | "history">("settings");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historySearch, setHistorySearch] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [selectedHistory, setSelectedHistory] = useState<HistoryItem | null>(null);
  const [historyDetailCopyNotice, setHistoryDetailCopyNotice] = useState("");
  const [isEditingHistoryTitle, setIsEditingHistoryTitle] = useState(false);
  const [historyTitleDraft, setHistoryTitleDraft] = useState("");
  const [savingHistoryTitle, setSavingHistoryTitle] = useState(false);
  const [activeNav, setActiveNav] = useState("generate");
  const theme = useSyncExternalStore(subscribeToTheme, getSavedTheme, getServerTheme);
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [accountProfileOpen, setAccountProfileOpen] = useState(false);
  const [showAuth, setShowAuth] = useState(false);
  const [registerMode, setRegisterMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [jobProgress, setJobProgress] = useState<GenerationJob | null>(null);
  const [streamedAudioChunks, setStreamedAudioChunks] = useState(0);
  const [authLoading, setAuthLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [audioName, setAudioName] = useState("");
  const [autoPlayHistoryAudio, setAutoPlayHistoryAudio] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [username, setUsername] = useState("");
  const [fullName, setFullName] = useState("");
  const [engineMode, setEngineMode] = useState<"local" | "server">("server");
  const [adapterStatus, setAdapterStatus] = useState<LocalAdapterStatus>({
    webGpuAvailable: false,
    adapterName: "",
    intelSelected: false,
  });
  const [modelStatusLocal, setModelStatusLocal] = useState<LocalModelStatus>({
    downloaded: false,
    downloading: false,
    totalBytes: 0,
    downloadedBytes: 0,
    percent: 0,
    speedBytesPerSecond: 0,
    error: "",
  });
  const [storageInfo, setStorageInfo] = useState({ usage: 0, quota: 0 });
  const audioRef = useRef<HTMLAudioElement>(null);
  const historyTitleInputRef = useRef<HTMLInputElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourcesRef = useRef(new Set<AudioBufferSourceNode>());
  const editorRef = useRef<HTMLTextAreaElement>(null);
  const accountMenuRef = useRef<HTMLDivElement>(null);
  const voiceMenuRef = useRef<HTMLDivElement>(null);
  const voiceTriggerRef = useRef<HTMLButtonElement>(null);
  const voiceIdRef = useRef("");
  const preferencesLoadedFor = useRef<string | null>(null);

  useEffect(() => {
    void getLocalAdapterStatus().then(setAdapterStatus);
    void isModelCached().then((downloaded) =>
      setModelStatusLocal((current) => ({ ...current, downloaded })),
    );
    void getModelStorageInfo().then(setStorageInfo);
    void requestPersistentStorage();
    if ("serviceWorker" in navigator) {
      void navigator.serviceWorker.register("/sw.js").catch(() => undefined);
    }
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (!accountMenuOpen) return;
    const closeOnOutsideClick = (event: PointerEvent) => {
      if (event.target instanceof Node && !accountMenuRef.current?.contains(event.target)) {
        setAccountMenuOpen(false);
        setAccountProfileOpen(false);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setAccountMenuOpen(false);
        setAccountProfileOpen(false);
      }
    };
    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [accountMenuOpen]);

  useEffect(() => {
    if (!voiceMenuOpen) return;
    const closeOnOutsideClick = (event: PointerEvent) => {
      if (event.target instanceof Node && !voiceMenuRef.current?.contains(event.target)) {
        setVoiceMenuOpen(false);
        setVoiceSearch("");
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setVoiceMenuOpen(false);
        setVoiceSearch("");
        voiceTriggerRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [voiceMenuOpen]);

  useEffect(() => {
    const textarea = editorRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    const maxHeight = Number.parseFloat(window.getComputedStyle(textarea).maxHeight);
    const contentHeight = textarea.scrollHeight;
    const nextHeight = Number.isFinite(maxHeight)
      ? Math.min(contentHeight, maxHeight)
      : contentHeight;
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = contentHeight > nextHeight ? "auto" : "hidden";
  }, [text]);

  useEffect(() => {
    let active = true;
    const refreshServerStatus = async () => {
      try {
        const response = await fetch(`${API_URL}/api/tts/status`);
        if (!response.ok) return;
        const status: ModelStatus = await response.json();
        if (active) setModelStatus(status);
      } catch {
        if (active) setModelStatus(null);
      }
    };
    void refreshServerStatus();
    const timer = window.setInterval(() => void refreshServerStatus(), 10000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const apiFetch = useCallback((path: string, init: RequestInit = {}, token?: string) => {
    const headers = new Headers(init.headers);
    if (token) headers.set("Authorization", `Bearer ${token}`);
    return fetch(`${API_URL}${path}`, { ...init, headers });
  }, []);

  const loadWorkspace = useCallback(async (current: Session): Promise<Voice[] | undefined> => {
    try {
      const [voiceResponse, historyResponse] = await Promise.all([
        apiFetch("/api/voices/", {}, current.token),
        apiFetch("/api/tts/history?page=1&limit=100", {}, current.token),
      ]);
      if (voiceResponse.status === 401 || historyResponse.status === 401) {
        localStorage.removeItem("kalapaktts-session");
        setSession(null);
        throw new Error("Your session expired. Please sign in again.");
      }
      if (!voiceResponse.ok) throw new Error(await responseError(voiceResponse));
      if (!historyResponse.ok) throw new Error(await responseError(historyResponse));
      const voiceList: Voice[] = await voiceResponse.json();
      const historyPage: HistoryPage = await historyResponse.json();
      setVoices(voiceList);
      setHistory(historyPage.items ?? []);
      setHistoryTotal(historyPage.total ?? historyPage.items?.length ?? 0);
      if (voiceList.length && !voiceIdRef.current) {
        const defaultVoice = voiceList.find((voice) => voice.name === "មាលី") ?? voiceList[0];
        voiceIdRef.current = defaultVoice.id;
        setVoiceId(defaultVoice.id);
      }
      return voiceList;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load your workspace.");
      return undefined;
    }
  }, [apiFetch]);

  const refreshHistory = useCallback(async () => {
    if (!session) {
      setShowAuth(true);
      return;
    }
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const response = await apiFetch("/api/tts/history?page=1&limit=100", {}, session.token);
      if (response.status === 401) {
        localStorage.removeItem("kalapaktts-session");
        setSession(null);
        throw new Error("Your session expired. Please sign in again.");
      }
      if (!response.ok) throw new Error(await responseError(response));
      const historyPage: HistoryPage = await response.json();
      setHistory(historyPage.items ?? []);
      setHistoryTotal(historyPage.total ?? historyPage.items?.length ?? 0);
    } catch (reason) {
      setHistoryError(reason instanceof Error ? reason.message : "Could not load your history.");
    } finally {
      setHistoryLoading(false);
    }
  }, [apiFetch, session]);

  const loadMoreHistory = useCallback(async () => {
    if (!session) {
      setShowAuth(true);
      return;
    }
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const nextPage = Math.floor(history.length / 100) + 1;
      const response = await apiFetch(`/api/tts/history?page=${nextPage}&limit=100`, {}, session.token);
      if (response.status === 401) {
        localStorage.removeItem("kalapaktts-session");
        setSession(null);
        throw new Error("Your session expired. Please sign in again.");
      }
      if (!response.ok) throw new Error(await responseError(response));
      const historyPage: HistoryPage = await response.json();
      setHistory((current) => [...current, ...historyPage.items]);
      setHistoryTotal(historyPage.total ?? history.length + historyPage.items.length);
    } catch (reason) {
      setHistoryError(reason instanceof Error ? reason.message : "Could not load older generations.");
    } finally {
      setHistoryLoading(false);
    }
  }, [apiFetch, history.length, session]);

  useEffect(() => {
    const stored = localStorage.getItem("kalapaktts-session");
    if (!stored) return;
    try {
      const parsed = JSON.parse(stored) as Session;
      if (parsed.token && parsed.user) {
        void Promise.resolve().then(() => {
          const preferences = readUserPreferences(parsed.user.id);
          setSpeed(preferences.speed);
          setSpeakingStyle(preferences.style);
          setPaceMode(preferences.paceMode);
          setPaceProfile(preferences.paceProfile);
          setTargetCps(preferences.targetCps);
          setPronunciationDictionary(preferences.pronunciationDictionary);
          setNoiseReduction(preferences.noiseReduction);
          preferencesLoadedFor.current = parsed.user.id;
          setSession(parsed);
          return loadWorkspace(parsed).then((voiceList) => {
            if (voiceList?.length) {
              const currentVoice = voiceList.find((voice) => voice.id === voiceIdRef.current) ?? voiceList[0];
              setInstruction(instructionForStyle(currentVoice, preferences.style));
            }
          });
        });
      }
    } catch {
      localStorage.removeItem("kalapaktts-session");
    }
  }, [loadWorkspace]);

  useEffect(() => {
    if (session) return;
    let active = true;
    const loadPublicVoices = async () => {
      const response = await apiFetch("/api/voices/");
      if (!response.ok) throw new Error(await responseError(response));
      const voiceList: Voice[] = await response.json();
      if (!active) return;
      setVoices(voiceList);
      if (voiceList.length && !voiceIdRef.current) {
        const defaultVoice = voiceList.find((voice) => voice.name === "មាលី") ?? voiceList[0];
        voiceIdRef.current = defaultVoice.id;
        setVoiceId(defaultVoice.id);
        setInstruction(defaultVoice.control_instruction ?? "");
      }
    };
    void loadPublicVoices().catch((reason) => {
      if (active) setError(reason instanceof Error ? reason.message : "Could not load public voices.");
    });
    return () => {
      active = false;
    };
  }, [apiFetch, session]);

  useEffect(() => {
    if (!session) {
      preferencesLoadedFor.current = null;
      return;
    }

    const userId = session.user.id;
    if (preferencesLoadedFor.current !== userId) return;
    const storageKey = `kalapaktts-preferences-${userId}`;
    localStorage.setItem(storageKey, JSON.stringify({
      pacePreferenceVersion: 2,
      noisePreferenceVersion: 2,
      speed,
      style: speakingStyle,
      paceMode,
      paceProfile,
      targetCps,
      pronunciationDictionary,
      noiseReduction,
    }));
  }, [session, speed, speakingStyle, paceMode, paceProfile, targetCps, pronunciationDictionary, noiseReduction]);

  useEffect(() => () => {
    if (audioUrl.startsWith("blob:")) URL.revokeObjectURL(audioUrl);
  }, [audioUrl]);

  useEffect(() => {
    if (!autoPlayHistoryAudio || activeNav !== "history" || !audioUrl || !audioRef.current) return;
    setAutoPlayHistoryAudio(false);
    void audioRef.current.play().catch(() => {
      setHistoryError("Automatic playback was blocked. Use the audio controls to start playback.");
    });
  }, [activeNav, audioUrl, autoPlayHistoryAudio]);

  useEffect(() => {
    if (!selectedHistory) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelectedHistory(null);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [selectedHistory]);

  useEffect(() => () => {
    audioContextRef.current?.close();
  }, []);

  const selectVoice = (id: string) => {
    voiceIdRef.current = id;
    setVoiceId(id);
    const voice = voices.find((item) => item.id === id);
    if (voice) setInstruction(instructionForStyle(voice, speakingStyle));
  };

  const selectStyle = (styleId: string) => {
    const style = speakingStyles.find((item) => item.id === styleId);
    setSpeakingStyle(styleId);
    if (style && "speed" in style) {
      setPaceMode("custom");
      setSpeed(style.speed);
    }
    if (style) setInstruction(instructionForStyle(voices.find((voice) => voice.id === voiceId), styleId));
  };

  const handleAuth = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setAuthLoading(true);
    setError("");
    try {
      if (registerMode) {
        const response = await apiFetch("/api/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, email, password, full_name: fullName || null }),
        });
        if (!response.ok) throw new Error(await responseError(response));
      }
      const form = new URLSearchParams({ username: email, password });
      const response = await apiFetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: form,
      });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json();
      const nextSession = { token: result.access_token, user: result.user as User };
      localStorage.setItem("kalapaktts-session", JSON.stringify(nextSession));
      const preferences = readUserPreferences(nextSession.user.id);
      setSpeed(preferences.speed);
      setSpeakingStyle(preferences.style);
      setPaceMode(preferences.paceMode);
      setPaceProfile(preferences.paceProfile);
      setTargetCps(preferences.targetCps);
      setPronunciationDictionary(preferences.pronunciationDictionary);
      setNoiseReduction(preferences.noiseReduction);
      preferencesLoadedFor.current = nextSession.user.id;
      setSession(nextSession);
      setShowAuth(false);
      setPassword("");
      const voiceList = await loadWorkspace(nextSession);
      if (voiceList?.length) {
        const currentVoice = voiceList.find((voice) => voice.id === voiceIdRef.current) ?? voiceList[0];
        setInstruction(instructionForStyle(currentVoice, preferences.style));
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not sign in.");
    } finally {
      setAuthLoading(false);
    }
  };

  const getAudio = async (url: string, name: string, autoPlay = false): Promise<boolean> => {
    if (!session) {
      setShowAuth(true);
      return false;
    }
    try {
      const response = await apiFetch(url, {}, session.token);
      if (!response.ok) throw new Error(await responseError(response));
      const blobUrl = URL.createObjectURL(await response.blob());
      setAudioUrl((oldUrl) => {
        if (oldUrl.startsWith("blob:")) URL.revokeObjectURL(oldUrl);
        return blobUrl;
      });
      setAudioName(name);
      setAutoPlayHistoryAudio(autoPlay);
      setError("");
      return true;
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "Audio could not be loaded.";
      setError(message);
      setHistoryError(message);
      return false;
    }
  };

  const downloadHistoryItem = async (item: HistoryItem) => {
    if (!session) {
      setShowAuth(true);
      return;
    }
    setHistoryError("");
    try {
      const response = await apiFetch(item.audio_url, {}, session.token);
      if (!response.ok) throw new Error(await responseError(response));
      const downloadUrl = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = `kalapaktts-${item.id}.wav`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
    } catch (reason) {
      setHistoryError(reason instanceof Error ? reason.message : "Audio download failed.");
    }
  };

  const openHistoryDetails = (item: HistoryItem) => {
    setSelectedHistory(item);
    setHistoryTitleDraft(item.title?.trim() || "Untitled");
    setIsEditingHistoryTitle(false);
    setHistoryError("");
    setHistoryDetailCopyNotice("");
    setAudioUrl("");
    setAudioName("");
    setAutoPlayHistoryAudio(false);
  };

  const saveHistoryTitle = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedHistory) return;
    if (!session) {
      setShowAuth(true);
      return;
    }

    const title = historyTitleDraft.trim();
    if (!title) {
      setHistoryError("Enter a title before saving.");
      historyTitleInputRef.current?.focus();
      return;
    }

    setSavingHistoryTitle(true);
    setHistoryError("");
    try {
      const response = await apiFetch(`/api/tts/history/${selectedHistory.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      }, session.token);
      if (response.status === 401) {
        localStorage.removeItem("kalapaktts-session");
        setSession(null);
        throw new Error("Your session expired. Please sign in again.");
      }
      if (!response.ok) throw new Error(await responseError(response));

      const updatedTitle: { title: string } = await response.json();
      const updatedHistory = { ...selectedHistory, title: updatedTitle.title };
      setSelectedHistory(updatedHistory);
      setHistory((current) => current.map((item) => item.id === updatedHistory.id ? updatedHistory : item));
      setHistoryTitleDraft(updatedTitle.title);
      setIsEditingHistoryTitle(false);
    } catch (reason) {
      setHistoryError(reason instanceof Error ? reason.message : "Could not save this title.");
    } finally {
      setSavingHistoryTitle(false);
    }
  };

  const copyHistoryText = async () => {
    if (!selectedHistory) return;
    try {
      await navigator.clipboard.writeText(selectedHistory.text_input);
      setHistoryDetailCopyNotice("Text copied.");
    } catch (reason) {
      setHistoryDetailCopyNotice(
        reason instanceof Error ? `Could not copy text: ${reason.message}` : "Could not copy text.",
      );
    }
  };

  const generate = async (requestedText: string = text, overrides?: GenerationOverrides) => {
    if (!requestedText.trim()) {
      setError("Enter some Khmer text to turn into speech.");
      editorRef.current?.focus();
      return;
    }
    if (engineMode === "local") {
      setError(
        "Local GPU synthesis is not active yet: a public independent ONNX decode loop for the 4-bit VoxCPM2 graphs has not been integrated. Server synthesis remains available.",
      );
      return;
    }
    if (!session) {
      setShowAuth(true);
      return;
    }
    setLoading(true);
    setModelStatus(null);
    setJobProgress(null);
    setStreamedAudioChunks(0);
    setError("");
    setNotice("");
    let monitoring = true;
    let statusTimer: number | undefined;
    try {
      const refreshModelStatus = async () => {
        const statusResponse = await apiFetch("/api/tts/status");
        if (!statusResponse.ok) throw new Error(await responseError(statusResponse));
        const status: ModelStatus = await statusResponse.json();
        if (monitoring) setModelStatus(status);
      };
      await refreshModelStatus();
      statusTimer = window.setInterval(() => {
        void refreshModelStatus().catch((reason: unknown) => {
          if (monitoring) setError(reason instanceof Error ? reason.message : "Could not check model download progress.");
        });
      }, 2000);
      for (const source of audioSourcesRef.current) source.stop();
      audioSourcesRef.current.clear();
      const audioContext = audioContextRef.current ?? new AudioContext();
      audioContextRef.current = audioContext;
      await audioContext.resume();
      const requestStartedAt = performance.now();
      let firstAudioReported = false;
      const response = await apiFetch("/api/tts/generate/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: requestedText,
          voice_preset_id: overrides ? overrides.voiceId ?? null : voiceId || null,
          control_instruction: (overrides ? overrides.instruction ?? "" : instruction).trim() || null,
          cfg_value: overrides?.guidance ?? guidance,
          inference_timesteps: overrides?.steps ?? steps,
          speed: overrides?.speed ?? speed,
          pace_mode: overrides?.paceMode ?? paceMode,
          target_cps: overrides ? overrides.targetCps ?? null : targetCps ?? paceProfileTargets.natural,
          pronunciation_dictionary: overrides?.pronunciationDictionary ?? pronunciationDictionary,
          noise_reduction: overrides?.noiseReduction ?? noiseReduction,
          style: overrides?.speakingStyle ?? speakingStyle,
        }),
      }, session.token);
      if (!response.ok) throw new Error(await responseError(response));
      if (!response.body) throw new Error("The browser could not open the speech audio stream.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      const generated: { value: NonNullable<GenerationJob["generation"]> | null } = { value: null };
      let nextPlaybackTime = audioContext.currentTime + 0.05;
      let streamFailure: string | null = null;
      const handleEvent = async (raw: string) => {
        const data = raw.split("\n").find((line) => line.startsWith("data: "));
        if (!data) return;
        const event = JSON.parse(data.slice(6)) as {
          type: "progress" | "audio" | "complete" | "error";
          status?: GenerationJob["status"];
          current_chunk?: number;
          total_chunks?: number;
          message?: string;
          queued_position?: number | null;
          audio_base64?: string;
          pause_ms?: number;
          generation?: NonNullable<GenerationJob["generation"]>;
          metrics?: GenerationJob["metrics"];
          error?: string;
        };
        if (event.type === "progress") {
          setJobProgress({
            job_id: "stream",
            status: event.status ?? "running",
            current_chunk: event.current_chunk ?? 0,
            total_chunks: event.total_chunks ?? 0,
            message: event.message ?? "Generating speech",
            queued_position: event.queued_position,
          });
        } else if (event.type === "audio" && event.audio_base64) {
          const binary = window.atob(event.audio_base64);
          const bytes = new Uint8Array(binary.length);
          for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
          const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
          const source = audioContext.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(audioContext.destination);
          source.onended = () => audioSourcesRef.current.delete(source);
          audioSourcesRef.current.add(source);
          const startAt = Math.max(nextPlaybackTime, audioContext.currentTime + 0.01);
          source.start(startAt);
          nextPlaybackTime = startAt + audioBuffer.duration + (event.pause_ms ?? 0) / 1000;
          setStreamedAudioChunks((count) => count + 1);
          if (!firstAudioReported) {
            firstAudioReported = true;
            setNotice(`First audio ready in ${((performance.now() - requestStartedAt) / 1000).toFixed(1)}s.`);
          }
        } else if (event.type === "complete") {
          generated.value = event.generation ?? null;
          if (event.metrics) {
            setLastGenerationMetrics(event.metrics);
            setLastMeasuredCps(event.metrics.speech_only_chars_per_second ?? null);
          }
        } else if (event.type === "error") {
          streamFailure = event.message ?? "Speech generation failed.";
        }
      };
      while (true) {
        const { value, done } = await reader.read();
        pending += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");
        const frames = pending.split("\n\n");
        pending = frames.pop() ?? "";
        for (const frame of frames) await handleEvent(frame);
        if (done) {
          if (pending.trim()) await handleEvent(pending);
          break;
        }
      }
      if (streamFailure) throw new Error(streamFailure);
      if (!generated.value) throw new Error("The speech stream ended without a saved audio file.");
      setJobProgress((current) => current ? { ...current, status: "completed", current_chunk: current.total_chunks, message: "Speech is ready" } : current);
      const audioLoaded = await getAudio(generated.value.audio_url, requestedText.slice(0, 72));
      if (!audioLoaded) return;
      const historyResponse = await apiFetch("/api/tts/history?page=1&limit=100", {}, session.token);
      if (historyResponse.ok) {
        const historyPage: HistoryPage = await historyResponse.json();
        setHistory(historyPage.items ?? []);
        setHistoryTotal(historyPage.total ?? historyPage.items?.length ?? 0);
      }
      setNotice("Your Khmer speech is ready.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Speech generation failed.");
    } finally {
      monitoring = false;
      if (statusTimer !== undefined) window.clearInterval(statusTimer);
      setLoading(false);
    }
  };

  const regenerateHistoryItem = (item: HistoryItem) => {
    setText(item.text_input);
    voiceIdRef.current = item.voice_id ?? "";
    setVoiceId(item.voice_id ?? "");
    if (item.control_instruction !== undefined) setInstruction(item.control_instruction ?? "");
    if (item.cfg_value !== undefined) setGuidance(item.cfg_value);
    if (item.inference_timesteps !== undefined) setSteps(item.inference_timesteps);
    setSpeed(item.speed);
    setPaceMode(item.pace_mode);
    setTargetCps(item.target_cps ?? null);
    setPronunciationDictionary(item.pronunciation_dictionary);
    setNoiseReduction(item.noise_reduction ?? false);
    setSpeakingStyle(item.style ?? "natural");
    setSelectedHistory(null);
    setActiveTab("settings");
    setActiveNav("generate");
    void generate(item.text_input, {
      voiceId: item.voice_id,
      instruction: item.control_instruction,
      guidance: item.cfg_value,
      steps: item.inference_timesteps,
      speed: item.speed,
      paceMode: item.pace_mode,
      targetCps: item.target_cps,
      pronunciationDictionary: item.pronunciation_dictionary,
      noiseReduction: item.noise_reduction,
      speakingStyle: item.style,
    });
  };

  const signOut = () => {
    localStorage.removeItem("kalapaktts-session");
    setSession(null);
    setHistory([]);
    setHistoryTotal(0);
    setHistorySearch("");
    setHistoryError("");
    setVoices([]);
    voiceIdRef.current = "";
    preferencesLoadedFor.current = null;
    setVoiceId("");
    setSpeed(1.0);
    setPaceMode("custom");
    setTargetCps(null);
    setPaceProfile("natural");
    setPronunciationDictionary(true);
    setSpeakingStyle("natural");
    setAudioUrl("");
    setNotice("");
  };

  const navItems = [
    { id: "generate", label: "Generate speech", icon: "sound" },
    { id: "history", label: "History", icon: "clock" },
  ];

  const toggleTheme = () => {
    saveTheme(theme === "dark" ? "light" : "dark");
  };

  const searchTerm = historySearch.trim().toLocaleLowerCase();
  const filteredHistory = history.filter((item) => {
    const voiceName = voices.find((voice) => voice.id === item.voice_id)?.name ?? "";
    return !searchTerm || `${item.text_input} ${voiceName}`.toLocaleLowerCase().includes(searchTerm);
  });
  const loadedDurations = history.flatMap((item) =>
    typeof item.audio_duration === "number" && Number.isFinite(item.audio_duration) ? [item.audio_duration] : [],
  );
  const loadedDurationTotal = loadedDurations.reduce((total, duration) => total + duration, 0);
  const thisMonthCount = history.filter((item) => {
    const created = new Date(item.created_at);
    const now = new Date();
    return created.getFullYear() === now.getFullYear() && created.getMonth() === now.getMonth();
  }).length;
  const selectedHistoryParts = selectedHistory
    ? selectedHistory.text_input.split(/(?<=[.!?។])\s+|\r?\n+/u).map((part) => part.trim()).filter(Boolean)
    : [];
  const selectedHistoryVoice = selectedHistory
    ? voices.find((voice) => voice.id === selectedHistory.voice_id)?.name ?? "Khmer voice"
    : "";

  const filteredVoices = voices.filter((voice) => {
    const query = voiceSearch.trim().toLowerCase();
    return !query || `${voice.name} ${voice.description ?? ""}`.toLowerCase().includes(query);
  });

  const clearEditor = () => {
    setText("");
    setAudioUrl("");
    setAudioName("");
    setLastMeasuredCps(null);
    setLastGenerationMetrics(null);
    setJobProgress(null);
    setStreamedAudioChunks(0);
    setError("");
    setNotice("");
    editorRef.current?.focus();
  };

  const offlineEngineStatus = !modelStatus
    ? "Checking local engine"
    : modelStatus.download_percent < 100
      ? "Downloading local model"
      : modelStatus.model_loaded
        ? "Offline-ready"
        : "Loading local model";
  const offlineEngineDetails = !modelStatus
    ? "Checking the local model and GPU status."
    : modelStatus.download_percent < 100
      ? `The one-time model download is ${modelStatus.download_percent}% complete.`
      : modelStatus.model_loaded && modelStatus.model_device === "cuda"
        ? `Runs locally on ${modelStatus.gpu_name ?? "this PC's GPU"} and works without internet after the one-time model download.`
        : modelStatus.model_loaded
          ? `Runs locally on this PC (${modelStatus.model_device ?? "CPU fallback"}) and works without internet after the one-time model download.`
          : "The model is downloaded locally and is loading.";
  let generationTitle = jobProgress?.message ?? "Preparing VoxCPM2 model";
  if (!jobProgress && modelStatus?.download_percent === 100) generationTitle = "Loading VoxCPM2 into memory";
  if (!jobProgress && modelStatus?.model_loaded) generationTitle = "Generating Khmer speech";
  let generateButtonLabel = "Generate speech";
  if (loading) {
    generateButtonLabel = jobProgress?.status === "queued"
      ? "Queued..."
      : modelStatus?.model_loaded
      ? jobProgress?.total_chunks && jobProgress.total_chunks > 1
        ? `Generating ${jobProgress.current_chunk}/${jobProgress.total_chunks}...`
        : "Generating..."
      : "Loading model...";
  }

  return (
    <main className="app-shell">
      <aside className="rail">
        <button className="brand-mark" aria-label="KalapakTTS home" onClick={() => { setActiveNav("generate"); setActiveTab("settings"); }}>
          <span className="brand-logo-crop">
            <Image className="brand-logo-black" src="/image/logo/Kalapak TTS logo black.png" alt="" width={1536} height={1024} />
          </span>
        </button>
        <div className="rail-links">
          {navItems.map((item) => (
            <button key={item.id} className={`rail-button ${activeNav === item.id ? "active" : ""}`} title={item.label} aria-label={item.label} onClick={() => {
              setActiveNav(item.id);
              if (item.id === "history") {
                setActiveTab("history");
                void refreshHistory();
              }
              if (item.id === "generate") setActiveTab("settings");
            }}>
              <Icon name={item.icon} size={19} />
            </button>
          ))}
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="topbar-title"><span>{activeNav === "history" ? "History" : "Generate Speech"}</span></div>
          <div className="topbar-actions">
            <div className="topbar-utilities">
              <a className="topbar-utility" title="VoxCPM2 project" aria-label="VoxCPM2 project" href="https://github.com/OpenBMB/VoxCPM" target="_blank" rel="noreferrer"><Icon name="globe" size={17} /></a>
              <button
                className="topbar-utility theme-toggle"
                type="button"
                title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
                aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
                aria-pressed={theme === "dark"}
                onClick={toggleTheme}
              >
                <Icon name={theme === "dark" ? "sun" : "moon"} size={17} />
              </button>
            </div>
            {session ? (
              <div className="account-menu-anchor" ref={accountMenuRef}>
                <button
                  className="avatar"
                  title="Account menu"
                  aria-label="Account menu"
                  aria-haspopup="menu"
                  aria-expanded={accountMenuOpen}
                  aria-controls="account-menu"
                  onClick={() => {
                    setAccountProfileOpen(false);
                    setAccountMenuOpen((open) => !open);
                  }}
                >
                  {(session.user.full_name || session.user.username).slice(0, 1).toUpperCase()}
                </button>
                {accountMenuOpen && (
                  <div className="account-menu" id="account-menu" role="menu" aria-label="Account menu">
                    {accountProfileOpen ? (
                      <>
                        <button className="account-menu-back" onClick={() => setAccountProfileOpen(false)}>← Account</button>
                        <strong className="account-menu-heading">My profile</strong>
                        <div className="account-profile-details">
                          <span><small>Name</small><strong>{session.user.full_name || session.user.username}</strong></span>
                          <span><small>Username</small><strong>@{session.user.username}</strong></span>
                          <span><small>Email</small><strong>{session.user.email}</strong></span>
                          <span><small>Role</small><strong>{session.user.role}</strong></span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="account-summary">
                          <span className="account-avatar">{(session.user.full_name || session.user.username).slice(0, 1).toUpperCase()}</span>
                          <span className="account-summary-copy">
                            <strong>{session.user.full_name || session.user.username}</strong>
                            <small>@{session.user.username}</small>
                            <small>{session.user.email}</small>
                          </span>
                        </div>
                        <div className="account-workspace">
                          <small>Current workspace</small>
                          <strong>Personal workspace</strong>
                          <span>{session.user.role}</span>
                        </div>
                        <div className="account-menu-items">
                          <button role="menuitem" onClick={() => setAccountProfileOpen(true)}><Icon name="user" size={17} />My profile</button>
                          <button role="menuitem" onClick={() => { setActiveTab("settings"); setActiveNav("generate"); setAccountMenuOpen(false); }}><Icon name="settings" size={17} />Settings</button>
                          <button role="menuitem" onClick={() => { setActiveTab("history"); setActiveNav("history"); setAccountMenuOpen(false); void refreshHistory(); }}><Icon name="clock" size={17} />History</button>
                        </div>
                        <div className="account-menu-separator" />
                        <div className="account-menu-items">
                          <button className="account-signout" role="menuitem" onClick={() => { setAccountMenuOpen(false); signOut(); }}><Icon name="logout" size={17} />Sign out</button>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            ) : <button className="sign-in-small" onClick={() => setShowAuth(true)}>Sign in</button>}
          </div>
        </header>

        <div className="main-grid">
          {activeNav === "history" ? (
            <section className="history-dashboard" aria-label="Generation history">
              <div className="history-dashboard-inner">
                <div className="history-page-header">
                  <div>
                    <span className="history-eyebrow">YOUR KHMER VOICE STUDIO</span>
                    <h1>History</h1>
                  </div>
                  <span className="history-category"><Icon name="sound" size={15} />Text to Speech</span>
                </div>
                <div className="history-stats">
                  <div className="history-stat"><span><Icon name="wave" size={15} />{historyTotal.toLocaleString()} generations</span></div>
                  <div className="history-stat"><span><Icon name="clock" size={15} />{formatHistoryDuration(loadedDurationTotal)} total</span><small>{history.length < historyTotal ? "Loaded records" : "Audio duration"}</small></div>
                  <div className="history-stat">
                    <span><Icon name="chart" size={15} />{formatHistoryDuration(loadedDurations.length ? loadedDurationTotal / loadedDurations.length : null)} avg</span>
                    <small>{loadedDurations.length ? `${loadedDurations.length} records` : "No audio yet"}</small>
                  </div>
                  <div className="history-stat"><span><Icon name="calendar" size={15} />{thisMonthCount} this month</span></div>
                </div>
                <div className="history-toolbar">
                  <label className="history-search">
                    <Icon name="search" size={17} />
                    <input
                      type="search"
                      aria-label="Search generations"
                      placeholder="Search generations..."
                      value={historySearch}
                      onChange={(event) => setHistorySearch(event.target.value)}
                    />
                  </label>
                  <div className="history-toolbar-actions">
                    <span>{history.length < historyTotal ? `${history.length} of ${historyTotal}` : `${filteredHistory.length} generations`}</span>
                    <button className="history-refresh" type="button" aria-label="Refresh history" disabled={historyLoading || !session} onClick={() => void refreshHistory()}>
                      {historyLoading ? <span className="spinner" /> : <Icon name="refresh" size={16} />}
                    </button>
                  </div>
                </div>
                {historyError && <div className="history-error" role="alert">{historyError}</div>}
                <div className="history-records">
                  {!session && <div className="history-empty"><strong>Sign in to view your history</strong><span>Your saved Khmer speech generations will appear here.</span><button onClick={() => setShowAuth(true)}>Sign in</button></div>}
                  {session && historyLoading && history.length === 0 && <div className="history-empty">Loading your history...</div>}
                  {session && !historyLoading && filteredHistory.length === 0 && (
                    <div className="history-empty">
                      <span className="empty-wave"><Icon name="wave" size={22} /></span>
                      <strong>{history.length ? "No generations match your search" : "No generations yet"}</strong>
                      <span>{history.length ? "Try a different search." : "Your Khmer speech will appear here after you generate it."}</span>
                    </div>
                  )}
                  {filteredHistory.map((item, index) => {
                    const voiceName = voices.find((voice) => voice.id === item.voice_id)?.name ?? "Khmer voice";
                    const createdAt = new Date(item.created_at);
                    return (
                      <article className="history-record" key={item.id}>
                        <button
                          className="history-record-play"
                          type="button"
                          aria-label={`Play generation ${index + 1}`}
                          onClick={() => void getAudio(item.audio_url, item.text_input.slice(0, 72), true)}
                        >
                          <Icon name="play" size={15} />
                        </button>
                        <button className="history-record-open" type="button" aria-label={`View details for generation ${index + 1}`} onClick={() => openHistoryDetails(item)}>
                          <div className="history-record-copy">
                            <div className="history-record-title"><strong>{item.title?.trim() || "Untitled"}</strong><span className={`history-status ${item.status === "failed" ? "failed" : ""}`}>{item.status ?? "Completed"}</span></div>
                            <p>{item.text_input}</p>
                            <small>
                              {voiceName} · {formatHistoryDuration(item.audio_duration)} · {item.text_input.length.toLocaleString()} characters ·
                              {" "}{Number.isNaN(createdAt.getTime()) ? item.created_at : createdAt.toLocaleDateString()}
                            </small>
                          </div>
                        </button>
                        <button
                          className="history-record-download"
                          type="button"
                          aria-label={`Download generation ${index + 1}`}
                          onClick={() => void downloadHistoryItem(item)}
                        >
                          <Icon name="download" size={17} />
                        </button>
                      </article>
                    );
                  })}
                </div>
                {audioUrl && !selectedHistory && (
                  <div className="history-audio-player">
                    <div><small>Now playing</small><strong>{audioName}</strong></div>
                    <audio ref={audioRef} controls src={audioUrl} />
                    <button type="button" aria-label="Close audio player" onClick={() => { setAudioUrl(""); setAudioName(""); }}>
                      <Icon name="close" size={16} />
                    </button>
                  </div>
                )}
                {session && history.length < historyTotal && (
                  <button className="history-load-more" type="button" disabled={historyLoading} onClick={() => void loadMoreHistory()}>
                    {historyLoading ? "Loading..." : "Load older generations"}
                  </button>
                )}
              </div>
            </section>
          ) : (
          <>
          <section className="editor-pane">
            <div className="editor-wrap">
              <div className="voice-picker" ref={voiceMenuRef}>
                <button
                  ref={voiceTriggerRef}
                  className="voice-picker-trigger"
                  type="button"
                  aria-label="Choose voice"
                  aria-haspopup="dialog"
                  aria-expanded={voiceMenuOpen}
                  aria-controls="voice-menu"
                  onClick={() => {
                    setVoiceSearch("");
                    setVoiceMenuOpen((open) => !open);
                  }}
                >
                  <span className="voice-orb">{voiceId ? "ម" : "ក"}</span>
                  <span className="voice-picker-name">{voices.find((voice) => voice.id === voiceId)?.name ?? "មាលី · Khmer"}</span>
                  <span className={`voice-picker-chevron ${voiceMenuOpen ? "open" : ""}`}><Icon name="chevron" size={16} /></span>
                </button>
                {voiceMenuOpen && (
                  <div className="voice-menu" id="voice-menu" role="dialog" aria-label="Choose voice">
                    <label className="voice-filter">
                      <Icon name="search" size={17} />
                      <input
                        type="search"
                        aria-label="Filter voices"
                        placeholder="Filter voices..."
                        value={voiceSearch}
                        onChange={(event) => setVoiceSearch(event.target.value)}
                        autoFocus
                      />
                    </label>
                    <div className="voice-menu-options" role="group" aria-label="Available voices">
                      {filteredVoices.length ? filteredVoices.map((voice) => (
                        <button
                          key={voice.id}
                          className={`voice-option ${voice.id === voiceId ? "selected" : ""}`}
                          type="button"
                          role="menuitemradio"
                          aria-checked={voice.id === voiceId}
                          data-value={voice.id}
                          onClick={() => {
                            selectVoice(voice.id);
                            setVoiceSearch("");
                            setVoiceMenuOpen(false);
                          }}
                        >
                          <span className="voice-option-orb">ម</span>
                          <span className="voice-option-copy">
                            <strong>{voice.name}</strong>
                            {voice.description && <small>{voice.description}</small>}
                          </span>
                          {voice.id === voiceId && <Icon name="check" size={17} />}
                        </button>
                      )) : <div className="voice-menu-empty">{voices.length ? "No voices match your search." : "No voices available."}</div>}
                    </div>
                  </div>
                )}
              </div>
              <textarea
                ref={editorRef}
                value={text}
                onChange={(event) => setText(event.target.value.slice(0, MAX_TEXT_LENGTH))}
                placeholder="Type or paste your text... សរសេរអត្ថបទជាភាសាខ្មែរនៅទីនេះ"
                maxLength={MAX_TEXT_LENGTH}
                aria-label="Text to convert to speech"
              />
              <div className="editor-hint"><span>Khmer · English supported</span><span>Type / for writing tips</span></div>
            </div>
            {(error || notice) && <div className={`feedback ${error ? "error" : "success"}`} role="status">{error || notice}</div>}
            {loading && <div className="generation-progress" role="status" aria-live="polite">
              <span className="spinner" />
              <span className="generation-progress-copy">
                <strong>{generationTitle}</strong>
                <small>{modelStatus && modelStatus.download_percent < 100
                  ? `First run downloads about 4.6 GB of model weights. Keep this page open · ${modelStatus.download_percent}% downloaded.`
                  : jobProgress?.total_chunks
                    ? jobProgress.status === "queued"
                      ? `Waiting for the speech-engine slot${jobProgress.queued_position ? ` · queue position ${jobProgress.queued_position}` : ""}`
                      : `Chunk ${jobProgress.current_chunk} of ${jobProgress.total_chunks} · Audio streams as each chunk is ready.`
                    : "Model files are ready. Loading the speech engine…"}
                </small>
                {modelStatus && modelStatus.download_percent < 100
                  ? <span className="progress-track"><span style={{ width: `${modelStatus.download_percent}%` }} /></span>
                  : jobProgress?.total_chunks && jobProgress.total_chunks > 1
                    ? <span className="progress-track"><span style={{ width: `${Math.min(100, jobProgress.current_chunk * 100 / jobProgress.total_chunks)}%` }} /></span>
                    : null}
                {streamedAudioChunks > 0 && <small className="streamed-audio-count">{streamedAudioChunks} text chunk{streamedAudioChunks === 1 ? "" : "s"} streamed and queued for playback</small>}
              </span>
            </div>}
            {audioUrl && <div className="audio-result">
              <div className="audio-result-title"><span className="audio-play-icon"><Icon name="play" size={15} /></span><span><strong>Speech preview</strong><small>{audioName}</small></span></div>
              <audio ref={audioRef} controls src={audioUrl} />
              <a className="download-link" href={audioUrl} download="kalapaktts-khmer.wav" aria-label="Download audio"><Icon name="download" size={16} /></a>
            </div>}
            {lastMeasuredCps !== null && <div className="feedback success last-pace-metric" aria-live="polite">
              <strong>Estimated speech pace · ល្បឿននិយាយប៉ាន់ស្មាន: {lastMeasuredCps.toFixed(1)} chars/s</strong>
              <small>
                Total: {lastGenerationMetrics?.total_chars_per_second?.toFixed(1) ?? "—"} chars/s ·
                {" "}{lastGenerationMetrics?.audio_duration_seconds?.toFixed(1) ?? "—"} s ·
                {" "}{lastGenerationMetrics?.pause_count_250ms ?? 0} pauses ≥250 ms
                {" "}({lastGenerationMetrics?.pause_seconds_250ms?.toFixed(1) ?? "0.0"} s total)
                {" "}· {lastGenerationMetrics?.pace_retry_count ?? 0} pace retries
              </small>
              <small>Estimated from audio activity; it is not aligned to individual words.</small>
            </div>}

            <footer className="editor-footer">
              <div className="editor-actions">
                <label className="import-button"><Icon name="document" size={17} />Import text<input type="file" accept=".txt,text/plain" onChange={async (event) => {
                  const file = event.target.files?.[0];
                  if (file) setText((await file.text()).slice(0, MAX_TEXT_LENGTH));
                  event.target.value = "";
                }} /></label>
                <button className="clear-button" type="button" disabled={loading} onClick={clearEditor}><Icon name="eraser" size={17} />Clear</button>
              </div>
              <div className="footer-right"><span className="counter">{text.length.toLocaleString()} / {MAX_TEXT_LENGTH.toLocaleString()}</span><button className="generate-button" disabled={loading || !text.trim()} onClick={() => void generate()}>{loading ? <span className="spinner" /> : <Icon name="sparkles" size={17} />}{generateButtonLabel}</button></div>
            </footer>
          </section>

          <aside className="settings-pane">
            <div className="tabs"><button className={activeTab === "settings" ? "selected" : ""} onClick={() => { setActiveTab("settings"); setActiveNav("generate"); }} >Settings</button><button className={activeTab === "history" ? "selected" : ""} onClick={() => {
              setActiveTab("history");
              setActiveNav("history");
              void refreshHistory();
            }}>History</button></div>
            {activeTab === "settings" ? (
              <div className="settings-scroll">
                <div className="section-label">Speech engine · ម៉ាស៊ីនសំឡេង</div>
                <div className="engine-card">
                  <label htmlFor="engine-mode">Engine selector</label>
                  <ModernDropdown
                    id="engine-mode"
                    label="Speech engine"
                    value={engineMode}
                    options={[
                      { value: "local", label: "Local GPU (offline) — Not available yet", disabled: true },
                      { value: "server", label: "This PC (offline)" },
                    ]}
                    onChange={(engine) => {
                      if (engine === "local" || engine === "server") setEngineMode(engine);
                    }}
                  />
                  <div className="engine-status engine-offline-status" role="status" aria-live="polite">
                    <strong>{offlineEngineStatus}</strong>
                    <small>{offlineEngineDetails}</small>
                  </div>
                  <div className="engine-status">
                    <strong>{adapterStatus.webGpuAvailable ? "WebGPU available" : "WebGPU unavailable"}</strong>
                    <small>{adapterStatus.adapterName || "Chrome/Edge WebGPU was not detected."}</small>
                  </div>
                  <div className="engine-status">
                    <strong>
                      This PC engine · {modelStatus?.model_device ?? modelStatus?.selected_device ?? modelStatus?.configured_device ?? "Checking"}
                      {modelStatus?.model_loaded ? " · warm" : ""}
                    </strong>
                    <small>
                      {modelStatus?.gpu_name ?? (modelStatus?.cuda_available ? "CUDA GPU detected" : "GPU status unavailable")}
                      {modelStatus?.torch_version ? ` · PyTorch ${modelStatus.torch_version}` : ""}
                      {modelStatus?.cuda_version ? ` · CUDA ${modelStatus.cuda_version}` : ""}
                      {modelStatus?.model_dtype ? ` · ${modelStatus.model_dtype}` : ""}
                      {modelStatus?.vae_device ? ` · AudioVAE ${modelStatus.vae_device}` : ""}
                    </small>
                    <small>
                      {modelStatus?.cuda_available
                        ? `VRAM used ${((modelStatus.vram_used_bytes ?? 0) / 1024 ** 3).toFixed(2)} / ${((modelStatus.vram_total_bytes ?? 0) / 1024 ** 3).toFixed(2)} GiB · model peak ${((modelStatus.peak_vram_bytes ?? 0) / 1024 ** 3).toFixed(2)} GiB`
                        : `CUDA BF16 ${modelStatus?.bf16_supported ? "available" : "unavailable"} · minimum free VRAM ${modelStatus?.min_free_vram_gib?.toFixed(1) ?? "5.8"} GiB`}
                    </small>
                    {modelStatus?.fallback_reason && <small className="engine-warning">
                      {modelStatus.fallback_reason}
                      {modelStatus.selected_device === "cpu" || modelStatus.model_device === "cpu" ? " Running on CPU." : ""}
                    </small>}
                  </div>
                  {adapterStatus.intelSelected && <small className="engine-warning">Intel/Intel UHD is selected. Open Windows Graphics settings and assign Chrome/Edge to the NVIDIA GPU with “High performance”.</small>}
                  <div className="engine-status">
                    <strong>{modelStatusLocal.downloaded ? "Model cached offline" : "Model not downloaded"}</strong>
                    <small>Uddom VoxCPM2 4-bit · about 1.94 GB · Apache-2.0</small>
                  </div>
                  {modelStatusLocal.downloading && (
                    <div className="local-model-progress">
                      <small>{modelStatusLocal.percent}% · {(modelStatusLocal.downloadedBytes / 1024 / 1024).toFixed(0)} MB / {(modelStatusLocal.totalBytes / 1024 / 1024).toFixed(0)} MB · {(modelStatusLocal.speedBytesPerSecond / 1024 / 1024).toFixed(1)} MB/s</small>
                      <span className="progress-track"><span style={{ width: `${modelStatusLocal.percent}%` }} /></span>
                    </div>
                  )}
                  {modelStatusLocal.error && <small className="engine-warning">{modelStatusLocal.error}</small>}
                  <div className="engine-actions">
                    <button
                      disabled={modelStatusLocal.downloading}
                      onClick={() => {
                        void downloadAndVerifyModel(setModelStatusLocal).catch((reason) => {
                          setModelStatusLocal((current) => ({
                            ...current,
                            downloading: false,
                            error: reason instanceof Error ? reason.message : "Model download failed.",
                          }));
                        });
                      }}
                    >
                      {modelStatusLocal.downloaded ? "Verify model" : "Download model"}
                    </button>
                    {modelStatusLocal.downloaded && <button onClick={() => {
                      void deleteCachedModel().then(async () => {
                        setModelStatusLocal((current) => ({ ...current, downloaded: false, downloadedBytes: 0, percent: 0, error: "" }));
                        setStorageInfo(await getModelStorageInfo());
                      });
                    }}>Delete model</button>}
                  </div>
                  <small className="engine-storage">
                    Storage used: {(storageInfo.usage / 1024 / 1024).toFixed(0)} MB
                    {storageInfo.quota ? ` · quota ${(storageInfo.quota / 1024 / 1024).toFixed(0)} MB` : ""}
                  </small>
                  <small className="engine-warning">Interrupted downloads are not resumable yet; complete files remain cached and are SHA-256 verified before reuse.</small>
                </div>
                <div className="section-label">Model</div>
                <div className="model-card"><span className="model-icon"><Icon name="sparkles" size={18} /></span><span><strong>VoxCPM2</strong><small>Multilingual · Khmer supported</small></span><Icon name="chevron" size={15} /></div>
                <div className="slider-card">
                  <label><span>Voice guidance</span><span className="slider-value">{guidance.toFixed(1)}</span></label>
                  <input aria-label="Voice guidance" type="range" min="0" max="10" step="0.1" value={guidance} onChange={(event) => setGuidance(Number(event.target.value))} />
                  <div className="slider-ends"><span>Expressive</span><span>Consistent</span></div>
                  <div className="thin-rule" />
                  <label htmlFor="quality-preset">Quality preset</label>
                  <ModernDropdown
                    id="quality-preset"
                    label="Quality preset"
                    value={qualityPreset}
                    options={[
                      { value: "fast", label: "Fast · 4 steps" },
                      { value: "balanced", label: "Balanced · 6 steps" },
                      { value: "high_quality", label: "High quality · 10 steps" },
                      { value: "custom", label: "Custom" },
                    ]}
                    onChange={(preset) => {
                      setQualityPreset(preset);
                      if (preset === "fast") setSteps(4);
                      if (preset === "balanced") setSteps(6);
                      if (preset === "high_quality") setSteps(10);
                    }}
                  />
                  <small className="pace-help">
                    High quality is the default for clearer speech; CPU fallback takes longer.
                    លំនាំដើមគុណភាពខ្ពស់អាចចំណាយពេលយូរនៅពេលប្រើ CPU។
                  </small>
                  <label><span>Generation quality</span><span className="slider-value">{steps} steps</span></label>
                  <input aria-label="Generation quality" type="range" min="1" max="30" step="1" value={steps} onChange={(event) => { setSteps(Number(event.target.value)); setQualityPreset("custom"); }} />
                  <div className="slider-ends"><span>Faster</span><span>More detail</span></div>
                  <div className="thin-rule" />
                  <label htmlFor="pace-profile">Pace · ល្បឿននិយាយ (speech-only)</label>
                  <ModernDropdown
                    id="pace-profile"
                    label="Pace profile"
                    value={paceProfile}
                    options={[
                      { value: "relaxed", label: "Relaxed · យឺតស្រួល (12 CPS)" },
                      { value: "natural", label: "Natural · ធម្មជាតិ (14 CPS)" },
                      { value: "brisk", label: "Brisk · រហ័សសមរម្យ (16 CPS)" },
                      { value: "custom", label: "Custom target · កំណត់ផ្ទាល់ខ្លួន" },
                    ]}
                    onChange={(profile) => {
                      if (profile === "relaxed" || profile === "natural" || profile === "brisk" || profile === "custom") {
                        setPaceProfile(profile);
                        setPaceMode("custom");
                        if (profile !== "custom") setTargetCps(paceProfileTargets[profile]);
                      }
                    }}
                  />
                  <label className="speed-label">
                    <span>Speaking speed · ល្បឿននិយាយ</span>
                    <span className="speed-current">{speed.toFixed(2)}x</span>
                    <button className="speed-reset" disabled={paceMode === "auto"} onClick={() => setSpeed(1.0)}>Reset</button>
                  </label>
                  <input aria-label="Speaking speed" type="range" min="0.7" max="1.5" step="0.05" value={speed} disabled={paceMode === "auto"} onChange={(event) => setSpeed(Number(event.target.value))} />
                  <div className="slider-ends"><span>Slower · យឺត</span><span>Faster · លឿន</span></div>
                  <div className="speed-presets" aria-label="Quick speaking speed presets">
                    {speedPresets.map((preset) => <button key={preset.value} className={speed === preset.value && paceMode === "custom" ? "selected" : ""} onClick={() => { setPaceMode("custom"); setSpeed(preset.value); }}>{preset.label}</button>)}
                  </div>
                  <div className="target-pace-row">
                    <label htmlFor="target-cps">Target pace · តួអក្សរ/វិនាទី</label>
                    <input id="target-cps" aria-label="Target pace" type="number" min="5" max="40" step="0.5" placeholder="14 (Natural default)" value={targetCps ?? ""} onChange={(event) => {
                      if (!event.target.value) {
                        setTargetCps(null);
                        setPaceProfile("custom");
                        return;
                      }
                      const value = Number(event.target.value);
                      if (Number.isFinite(value) && value >= 5 && value <= 40) {
                        setTargetCps(value);
                        setPaceProfile(
                          value === 12 ? "relaxed" : value === 14 ? "natural" : value === 16 ? "brisk" : "custom",
                        );
                      }
                    }} />
                  </div>
                  <small className="pace-help">
                    Absolute speech-only target; pace is estimated from the full audio after silence trimming. Pitch-preserving correction uses 0.85x–1.15x per chunk when needed. The 12/14/16 CPS profile values are targets; measured speech pace may vary by text.
                  </small>
                </div>
                <label className="dictionary-toggle">
                  <input type="checkbox" checked={pronunciationDictionary} onChange={(event) => setPronunciationDictionary(event.target.checked)} />
                  <span><strong>Khmer pronunciation dictionary</strong><small>Full-Stack, Portfolio, and Source Code are read with Khmer pronunciations.</small></span>
                </label>
                <label className="dictionary-toggle">
                  <input aria-label="Noise reduction" type="checkbox" checked={noiseReduction} onChange={(event) => setNoiseReduction(event.target.checked)} />
                  <span><strong>Noise reduction · កាត់បន្ថយសំឡេងរំខាន</strong><small>Applies gentle noise reduction after generation. It is off by default and can slightly soften speech. ប្រើការកាត់បន្ថយសំឡេងរំខានកម្រិតស្រាល ហើយបិទជាលំនាំដើម។</small></span>
                </label>
                <div className="section-label">Voice style</div>
                <div className="voice-style-card">
                  <div className="voice-style-heading"><span className="voice-orb small">ម</span><span><strong>{voices.find((voice) => voice.id === voiceId)?.name ?? "មាលី"}</strong><small>{speakingStyles.find((style) => style.id === speakingStyle)?.label}</small></span></div>
                  <label htmlFor="speaking-style">Speaking style · រចនាប័ទ្មសំឡេង</label>
                  <ModernDropdown
                    id="speaking-style"
                    label="Speaking style"
                    value={speakingStyle}
                    options={speakingStyles.map((style) => ({ value: style.id, label: style.label }))}
                    onChange={selectStyle}
                  />
                  <label htmlFor="voice-instruction">Describe how the voice should sound · ពិពណ៌នាសំឡេង</label>
                  <textarea id="voice-instruction" maxLength={1000} value={instruction} onChange={(event) => setInstruction(event.target.value)} placeholder="e.g. Warm, calm and clear Khmer narration..." />
                  <div className="instruction-count">{instruction.length}/1000</div>
                  <button className="style-preview-button" disabled={loading} onClick={() => void generate(styleSample)}>Preview style · ស្តាប់គំរូ</button>
                </div>
                <div className="section-label">Global settings</div>
                <div className="global-card">
                  <div className="setting-row"><span>Language</span><span className="setting-value"><Icon name="globe" size={15} /> Khmer <Icon name="chevron" size={13} /></span></div>
                  <div className="setting-row"><span>Generation mode</span><span className="setting-value">Standard <Icon name="chevron" size={13} /></span></div>
                  <div className="setting-row"><span>Model output</span><span className="setting-value">WAV · 48 kHz</span></div>
                </div>
                {!session && <div className="sign-in-note"><strong>Save your speech history</strong><span>Sign in to generate audio and keep your creations.</span><button onClick={() => setShowAuth(true)}>Sign in or create account <span>→</span></button></div>}
              </div>
            ) : (
              <div className="history-list">
                <div className="history-heading"><span>Recent generations</span><button onClick={() => void refreshHistory()}>Refresh</button></div>
                {!session && <div className="history-empty">Sign in to view your speech history.</div>}
                {session && history.length === 0 && <div className="history-empty"><span className="empty-wave"><Icon name="wave" size={22} /></span><strong>No generations yet</strong><span>Your Khmer speech will appear here.</span></div>}
                {history.map((item) => <button className="history-item" key={item.id} onClick={() => getAudio(item.audio_url, item.text_input.slice(0, 72))}>
                  <span className="history-play"><Icon name="play" size={14} /></span><span className="history-copy"><strong>{item.text_input}</strong><small>{new Date(item.created_at).toLocaleString()} · {item.audio_duration ? `${item.audio_duration.toFixed(1)} sec` : "Khmer speech"} · {item.pace_mode === "custom" ? `${item.speed.toFixed(2)}x custom` : "Auto pace"} · {typeof item.target_cps === "number" ? `${item.target_cps.toFixed(1)} cps target` : "target not recorded"}{item.pronunciation_dictionary ? " · Khmer dictionary" : ""}{item.noise_reduction ? " · Noise reduction" : ""} · {speakingStyles.find((style) => style.id === item.style)?.label ?? item.style ?? "Custom"}</small></span>
                </button>)}
              </div>
            )}
          </aside>
          </>
          )}
        </div>
      </section>

      {selectedHistory && (
        <div
          className="history-detail-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setSelectedHistory(null);
          }}
        >
          <section
            className="history-detail-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby={isEditingHistoryTitle ? undefined : "history-detail-title"}
            aria-label={isEditingHistoryTitle ? "Edit generation title" : undefined}
          >
            <div className="history-detail-script">
              <div className="history-detail-script-header">
                <strong>Script · {selectedHistoryParts.length} {selectedHistoryParts.length === 1 ? "part" : "parts"}</strong>
                <button type="button" onClick={() => void copyHistoryText()}><Icon name="copy" size={15} />Copy</button>
              </div>
              <div className="history-detail-parts">
                {selectedHistoryParts.map((part, index) => (
                  <div className="history-detail-part" key={`${index}-${part.slice(0, 20)}`}>
                    <span>{index + 1}</span><p>{part}</p>
                  </div>
                ))}
              </div>
              {historyDetailCopyNotice && <small className="history-copy-notice" role="status">{historyDetailCopyNotice}</small>}
              <div className="history-detail-player">
                {!audioUrl && (
                  <button type="button" onClick={() => void getAudio(selectedHistory.audio_url, selectedHistory.text_input.slice(0, 72), true)}>
                    <Icon name="play" size={15} />Play audio
                  </button>
                )}
                {audioUrl && <audio ref={audioRef} controls src={audioUrl} />}
              </div>
            </div>
            <aside className="history-detail-info">
              <div className="history-detail-info-top">
                <span>{new Date(selectedHistory.created_at).toLocaleDateString()}</span>
                <button type="button" aria-label="Close generation details" onClick={() => setSelectedHistory(null)}><Icon name="close" size={17} /></button>
              </div>
              <div className="history-detail-title-row">
                {isEditingHistoryTitle ? (
                  <form className="history-detail-title-form" onSubmit={(event) => void saveHistoryTitle(event)}>
                    <input
                      ref={historyTitleInputRef}
                      aria-label="Generation title"
                      autoFocus
                      maxLength={100}
                      value={historyTitleDraft}
                      onChange={(event) => setHistoryTitleDraft(event.target.value)}
                      disabled={savingHistoryTitle}
                    />
                    <button className="history-title-action save" type="submit" aria-label="Save title" disabled={savingHistoryTitle}>
                      {savingHistoryTitle ? <span className="spinner" /> : <Icon name="check" size={16} />}
                    </button>
                    <button className="history-title-action" type="button" aria-label="Cancel title edit" onClick={() => {
                      setHistoryTitleDraft(selectedHistory.title?.trim() || "Untitled");
                      setIsEditingHistoryTitle(false);
                      setHistoryError("");
                    }} disabled={savingHistoryTitle}>
                      <Icon name="close" size={15} />
                    </button>
                  </form>
                ) : (
                  <>
                    <h2 id="history-detail-title">{selectedHistory.title?.trim() || "Untitled"}</h2>
                    <button className="history-title-edit" type="button" aria-label="Edit title" onClick={() => {
                      setHistoryTitleDraft(selectedHistory.title?.trim() || "Untitled");
                      setIsEditingHistoryTitle(true);
                      setHistoryError("");
                    }}>
                      <Icon name="edit" size={15} />
                    </button>
                  </>
                )}
              </div>
              <p className="history-detail-date">
                {Number.isNaN(new Date(selectedHistory.created_at).getTime())
                  ? selectedHistory.created_at
                  : new Date(selectedHistory.created_at).toLocaleString()}
              </p>
              <div className="history-detail-metadata">
                <div><span>Voice</span><strong>{selectedHistoryVoice}</strong></div>
                <div><span>Type</span><strong>{selectedHistoryParts.length > 1 ? "Multi-part" : "Single"}</strong></div>
                <div><span>Length</span><strong>{formatHistoryDuration(selectedHistory.audio_duration)}</strong></div>
                <div><span>Characters</span><strong>{selectedHistory.text_input.length.toLocaleString()}</strong></div>
                <div><span>Status</span><strong className={`history-status ${selectedHistory.status === "failed" ? "failed" : ""}`}>{selectedHistory.status ?? "Completed"}</strong></div>
              </div>
              <div className="history-detail-actions">
                <button
                  className="history-detail-regenerate"
                  type="button"
                  onClick={() => regenerateHistoryItem(selectedHistory)}
                  disabled={loading}
                >
                  <Icon name="refresh" size={16} />Regenerate speech
                </button>
                <button className="history-detail-download" type="button" onClick={() => void downloadHistoryItem(selectedHistory)}>
                  <Icon name="download" size={16} />Download audio
                </button>
              </div>
              {historyError && <div className="history-error" role="alert">{historyError}</div>}
            </aside>
          </section>
        </div>
      )}

      {showAuth && <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setShowAuth(false); }}>
        <section className="auth-modal" role="dialog" aria-modal="true" aria-labelledby="auth-title">
          <button className="modal-close" aria-label="Close" onClick={() => setShowAuth(false)}><Icon name="close" /></button>
          <div className="modal-brand brand-logo-crop">
            <Image className="brand-logo-white" src="/image/logo/Kalapak TTS white.png" alt="" width={1672} height={941} />
          </div>
          <p className="eyebrow">YOUR KHMER VOICE STUDIO</p>
          <h2 id="auth-title">{registerMode ? "Create your account" : "Welcome back"}</h2>
          <p className="modal-copy">{registerMode ? "Join KalapakTTS and bring your words to life." : "Sign in to generate speech and access your history."}</p>
          <form onSubmit={handleAuth}>
            {registerMode && <>
              <label htmlFor="name">Full name <span>(optional)</span></label><input id="name" autoComplete="name" value={fullName} onChange={(event) => setFullName(event.target.value)} />
              <label htmlFor="username">Username</label><input id="username" required minLength={3} autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} />
            </>}
            <label htmlFor="email">Email address</label><input id="email" type="email" required autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} />
            <label htmlFor="password">Password</label><input id="password" type="password" required minLength={6} autoComplete={registerMode ? "new-password" : "current-password"} value={password} onChange={(event) => setPassword(event.target.value)} />
            {error && <div className="auth-error" role="alert">{error}</div>}
            <button className="auth-submit" disabled={authLoading}>{authLoading ? "Please wait..." : registerMode ? "Create account" : "Sign in"}<span>→</span></button>
          </form>
          <div className="auth-switch">{registerMode ? "Already have an account?" : "New to KalapakTTS?"}<button onClick={() => { setRegisterMode(!registerMode); setError(""); }}>{registerMode ? "Sign in" : "Create account"}</button></div>
          <div className="khmer-auth-copy">សំឡេងខ្មែររបស់អ្នក ចាប់ផ្តើមនៅទីនេះ</div>
        </section>
      </div>}
    </main>
  );
}
