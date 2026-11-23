from html.parser import HTMLParser
import re
import unittest
from pathlib import Path


UI_ROOT = Path(__file__).resolve().parents[1] / "ui"
RUNTIME_ASSET_SUFFIXES = {".html", ".js", ".css", ".json", ".txt", ".webmanifest"}
EXTERNAL_URL_PATTERN = re.compile(r"https?://[^\s\"'`<>]+", re.IGNORECASE)
CSS_EXTERNAL_URL_PATTERN = re.compile(
    r"""(?is)(?:url\(\s*['"]?\s*https?://|@import\s+(?:url\()?['"]?\s*https?://)"""
)
JS_NETWORK_URL_PATTERNS = (
    re.compile(r"""\bfetch\s*\(\s*['"]\s*https?://""", re.IGNORECASE),
    re.compile(r"""\bnew\s+(?:WebSocket|EventSource)\s*\(\s*['"]\s*https?://""", re.IGNORECASE),
    re.compile(r"""\.open\s*\(\s*['"][A-Z]+['"]\s*,\s*['"]\s*https?://""", re.IGNORECASE),
    re.compile(r"""\bimport\s*\(\s*['"]\s*https?://""", re.IGNORECASE),
    re.compile(r"""\b(?:src|href)\s*=\s*['"]\s*https?://""", re.IGNORECASE),
    re.compile(r"""\.setAttribute\(\s*['"](?:src|href)['"]\s*,\s*['"]\s*https?://""", re.IGNORECASE),
    re.compile(r"""\b(?:window\.)?open\s*\(\s*['"]\s*https?://""", re.IGNORECASE),
    re.compile(r"""\b(?:window\.)?location(?:\.href)?\s*=\s*['"]\s*https?://""", re.IGNORECASE),
)


class ExternalMarkupUrlParser(HTMLParser):
    URL_ATTRIBUTES = {
        "a": {"href"},
        "area": {"href"},
        "audio": {"src"},
        "base": {"href"},
        "embed": {"src"},
        "form": {"action"},
        "frame": {"src"},
        "iframe": {"src"},
        "img": {"src", "srcset"},
        "link": {"href"},
        "object": {"data"},
        "script": {"src"},
        "source": {"src", "srcset"},
        "track": {"src"},
        "video": {"poster", "src"},
    }

    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in self.URL_ATTRIBUTES.get(_tag.lower(), set()) and value:
                self.urls.extend(EXTERNAL_URL_PATTERN.findall(value))


class DesktopUiBuildTests(unittest.TestCase):
    def test_export_contains_no_external_runtime_urls_or_browser_tts_engine(self) -> None:
        runtime_files = [
            path
            for path in UI_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in RUNTIME_ASSET_SUFFIXES
        ]
        self.assertTrue(runtime_files, "Build the desktop static UI before running this test.")

        violations: list[str] = []
        runtime_text = ""
        for path in runtime_files:
            content = path.read_text(encoding="utf-8", errors="replace")
            runtime_text += content
            relative_path = str(path.relative_to(UI_ROOT))
            if path.suffix.lower() == ".html":
                parser = ExternalMarkupUrlParser()
                parser.feed(content)
                violations.extend(f"{relative_path}: {url}" for url in parser.urls)
            elif path.suffix.lower() == ".css":
                if CSS_EXTERNAL_URL_PATTERN.search(content):
                    violations.append(f"{relative_path}: external CSS resource URL")
            elif path.suffix.lower() == ".js":
                if any(pattern.search(content) for pattern in JS_NETWORK_URL_PATTERNS):
                    violations.append(f"{relative_path}: external JavaScript network URL")
            elif path.suffix.lower() in {".json", ".webmanifest"}:
                violations.extend(
                    f"{relative_path}: {url}" for url in EXTERNAL_URL_PATTERN.findall(content)
                )

        self.assertEqual(violations, [], "External URL found in runtime assets:\n" + "\n".join(violations))
        for forbidden in (
            "Local GPU (browser)",
            "onnxruntime",
            "navigator.gpu",
            "webgpu",
            "onnx",
            "navigator.serviceworker",
            "serviceworker.register",
            "huggingface.co",
            "kalapaktts-session",
            "/api/auth/",
            "Create your account",
        ):
            self.assertNotIn(forbidden.casefold(), runtime_text.casefold())

    def test_export_uses_only_local_font_assets(self) -> None:
        font_files = list((UI_ROOT / "fonts").glob("*.woff2"))
        self.assertTrue(font_files)
        css_files = list(UI_ROOT.rglob("*.css"))
        css_text = "\n".join(path.read_text(encoding="utf-8") for path in css_files)
        self.assertIn("/fonts/kantumruy-khmer-normal.woff2", css_text)
        self.assertIsNone(CSS_EXTERNAL_URL_PATTERN.search(css_text))

    def test_export_bundles_the_local_kalapak_brand_mark(self) -> None:
        logo_file = UI_ROOT / "branding" / "kalapak-mark-white.png"
        self.assertTrue(logo_file.is_file(), "The KalapakTTS brand mark must be bundled locally.")

        runtime_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in UI_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in RUNTIME_ASSET_SUFFIXES
        )
        self.assertIn("kalapak-mark-white.png", runtime_text)


if __name__ == "__main__":
    unittest.main()
