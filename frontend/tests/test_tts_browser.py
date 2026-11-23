import json
import re
import os
import subprocess
import unittest
import uuid
import wave
import io
import time
from pathlib import Path
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright


BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
API_URL = os.environ.get("KALAPAK_API_URL", "http://127.0.0.1:8000")


def api_request(method, path, token):
    request = Request(
        f"{API_URL}{path}",
        headers={"Authorization": f"Bearer {token}"},
        method=method,
    )
    with urlopen(request, timeout=30) as response:
        return response.status, response.headers, response.read()


def wait_for_generation_job(job_id, token, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status_code, _, body = api_request(
            "GET",
            f"/api/tts/generate/jobs/{job_id}",
            token,
        )
        if status_code != 200:
            raise AssertionError(f"Job polling failed ({status_code}): {body!r}")
        job = json.loads(body)
        if job["status"] == "completed":
            return job
        if job["status"] == "failed":
            raise AssertionError(f"Speech job failed: {job['error']}")
        time.sleep(1)
    raise AssertionError(f"Speech job {job_id} did not finish within {timeout} seconds")


def select_dropdown_option(page, label, value):
    page.get_by_role("combobox", name=label).click()
    page.get_by_role("listbox", name=label).locator(f'[data-value="{value}"]').click()


def select_voice(page, voice_id):
    page.get_by_role("button", name="Choose voice").click()
    page.get_by_role("dialog", name="Choose voice").locator(f'[data-value="{voice_id}"]').click()


class KhmerSpeechBrowserTests(unittest.TestCase):
    def test_legacy_fast_pace_preferences_migrate_to_natural(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.add_init_script(
                    "localStorage.setItem('kalapaktts-session', JSON.stringify({"
                    "token:'browser-test-token',user:{id:'pace-migration',username:'pace-test',"
                    "email:'pace-test@example.com',full_name:'Pace Test',role:'user'}}));"
                    "localStorage.setItem('kalapaktts-preferences-pace-migration', JSON.stringify({"
                    "speed:1,style:'natural',paceMode:'custom',paceProfile:'custom',targetCps:18,"
                    "pacePreferenceVersion:1,pronunciationDictionary:true,noiseReduction:false,"
                    "noisePreferenceVersion:2}));"
                )

                def fulfill_workspace(route):
                    if "history" in route.request.url:
                        body = {"items": [], "total": 0}
                    elif "voices" in route.request.url:
                        body = [{
                            "id": "mali",
                            "name": "មាលី",
                            "description": "Natural Khmer voice",
                            "control_instruction": "",
                            "has_reference_audio": False,
                        }]
                    else:
                        body = {}
                    route.fulfill(status=200, content_type="application/json", body=json.dumps(body))

                page.route(
                    re.compile(r"http://localhost:8000/api/(voices/|tts/history|tts/status)"),
                    fulfill_workspace,
                )
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                page.wait_for_function(
                    "() => document.querySelector('#target-cps')?.value === '14'"
                )

                self.assertEqual(page.locator("#target-cps").input_value(), "14")
                self.assertEqual(
                    page.get_by_role("combobox", name="Pace profile").get_attribute("data-value"),
                    "natural",
                )
                migrated = page.evaluate(
                    "() => JSON.parse(localStorage.getItem('kalapaktts-preferences-pace-migration'))"
                )
                self.assertEqual(migrated["pacePreferenceVersion"], 2)
                self.assertEqual(migrated["targetCps"], 14)
                self.assertEqual(migrated["paceProfile"], "natural")
            finally:
                browser.close()

    def test_account_avatar_opens_working_profile_menu(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.add_init_script(
                    "localStorage.setItem('kalapaktts-session', JSON.stringify({"
                    "token:'browser-test-token',user:{id:'browser-test',username:'browser-test',"
                    "email:'browser-test@example.com',full_name:'Browser Test',role:'user'}}));"
                )
                page.route(
                    re.compile(r"http://localhost:8000/api/(voices/|tts/history)"),
                    lambda route: route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps({"items": []} if "history" in route.request.url else []),
                    ),
                )
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                self.assertEqual(page.get_by_role("button", name="Account", exact=True).count(), 0)
                menu = page.get_by_role("menu", name="Account menu")
                avatar = page.get_by_role("button", name="Account menu")
                avatar.click()
                self.assertTrue(menu.is_visible())
                self.assertIn("Personal workspace", menu.inner_text())
                self.assertIn("browser-test@example.com", menu.inner_text())

                page.get_by_role("menuitem", name="My profile").click()
                self.assertTrue(page.get_by_text("My profile", exact=True).is_visible())
                self.assertTrue(page.get_by_text("browser-test@example.com", exact=True).is_visible())
                page.get_by_role("button", name="← Account").click()

                page.get_by_role("menuitem", name="Settings").click()
                self.assertFalse(menu.is_visible())
                avatar.click()
                page.get_by_role("menuitem", name="History").click()
                self.assertTrue(page.locator(".history-dashboard").is_visible())
                avatar.click()
                page.get_by_role("menuitem", name="Sign out").click()
                self.assertTrue(page.locator(".topbar").get_by_role("button", name="Sign in", exact=True).is_visible())
            finally:
                browser.close()

    def test_editor_uses_horizontal_space_at_desktop_widths(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1080, "height": 768})
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                layout = page.evaluate(
                    "() => ({ "
                    "editorPadding: parseFloat(getComputedStyle(document.querySelector('.editor-pane')).paddingLeft), "
                    "editorWidth: document.querySelector('.editor-wrap').getBoundingClientRect().width, "
                    "settingsLeft: document.querySelector('.settings-pane').getBoundingClientRect().left, "
                    "viewportWidth: document.documentElement.clientWidth "
                    "})"
                )
                self.assertLessEqual(layout["editorPadding"], 55)
                self.assertGreater(layout["editorWidth"], 450)
                self.assertLess(layout["settingsLeft"], layout["viewportWidth"])
            finally:
                browser.close()

    def test_generated_editor_content_scrolls_without_shrinking_text_area(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 966, "height": 768})
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                layout = page.evaluate(
                    """() => {
                      const pane = document.querySelector('.editor-pane');
                      const editor = document.querySelector('.editor-wrap');
                      const footer = document.querySelector('.editor-footer');
                      const editorHeight = editor.getBoundingClientRect().height;
                      for (const [className, height] of [
                        ['feedback success', 48],
                        ['audio-result', 145],
                        ['feedback success last-pace-metric', 90],
                      ]) {
                        const result = document.createElement('div');
                        result.className = className;
                        result.style.height = `${height}px`;
                        pane.insertBefore(result, footer);
                      }
                      return {
                        overflowY: getComputedStyle(pane).overflowY,
                        editorFlexShrink: getComputedStyle(editor).flexShrink,
                        editorHeight,
                        editorHeightAfter: editor.getBoundingClientRect().height,
                        paneScrolls: pane.scrollHeight > pane.clientHeight,
                      };
                    }"""
                )
                self.assertEqual(layout["overflowY"], "auto")
                self.assertEqual(layout["editorFlexShrink"], "0")
                self.assertEqual(layout["editorHeightAfter"], layout["editorHeight"])
                self.assertTrue(layout["paneScrolls"])
            finally:
                browser.close()

    def test_desktop_editor_actions_follow_the_text_box(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                spacing = page.evaluate(
                    "() => document.querySelector('.editor-footer').getBoundingClientRect().top "
                    "- document.querySelector('.editor-wrap').getBoundingClientRect().bottom"
                )
                self.assertGreaterEqual(spacing, 10)
                self.assertLessEqual(spacing, 22)
            finally:
                browser.close()

    def test_speech_pace_metrics_are_stacked_and_readable(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1080, "height": 768})
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                metrics = page.evaluate(
                    """() => {
                      const panel = document.createElement('div');
                      panel.className = 'feedback success last-pace-metric';
                      panel.innerHTML = '<strong>Estimated speech pace</strong>'
                        + '<small>Total: 16.2 chars/s · 145.5 s · 12 pauses</small>'
                        + '<small>Estimated from audio activity.</small>';
                      document.body.append(panel);
                      return [...panel.querySelectorAll('strong, small')].map((element) => ({
                        display: getComputedStyle(element).display,
                        top: element.getBoundingClientRect().top,
                      }));
                    }"""
                )
                self.assertEqual([metric["display"] for metric in metrics], ["block"] * 3)
                self.assertLess(metrics[0]["top"], metrics[1]["top"])
                self.assertLess(metrics[1]["top"], metrics[2]["top"])
            finally:
                browser.close()

    def test_desktop_scrollable_content_has_space_from_scrollbar(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                padding_right = page.locator(".settings-scroll").evaluate(
                    "(element) => parseFloat(getComputedStyle(element).paddingRight)"
                )
                self.assertGreaterEqual(padding_right, 12)
                page.locator(".tabs").get_by_role("button", name="History").click()
                history_styles = page.locator(".history-list").evaluate(
                    "(element) => ({ "
                    "paddingRight: parseFloat(getComputedStyle(element).paddingRight), "
                    "scrollbarGutter: getComputedStyle(element).scrollbarGutter "
                    "})"
                )
                self.assertGreaterEqual(history_styles["paddingRight"], 12)
                self.assertEqual(history_styles["scrollbarGutter"], "stable")
            finally:
                browser.close()

    def test_settings_quick_actions_are_removed(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                self.assertEqual(page.locator(".settings-quick-actions").count(), 0)
                for label in ("Auto style", "Voice library", "Clear"):
                    self.assertEqual(page.get_by_role("button", name=label).count(), 0)
                self.assertTrue(page.get_by_text("Speech engine", exact=False).is_visible())
            finally:
                browser.close()

    def test_signed_in_account_banner_is_removed(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.add_init_script(
                    "localStorage.setItem('kalapaktts-session', JSON.stringify({"
                    "token:'browser-test-token',user:{id:'browser-test',username:'browser-test',"
                    "email:'browser-test@example.com',role:'user'}}));"
                )
                page.route(
                    re.compile(r"http://localhost:8000/api/(voices/|tts/history)"),
                    lambda route: route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps({"items": []} if "history" in route.request.url else []),
                    ),
                )
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                page.wait_for_function(
                    "() => JSON.parse(localStorage.getItem('kalapaktts-session'))?.user?.username === 'browser-test'"
                )
                self.assertEqual(page.locator(".account-note").count(), 0)
                self.assertNotIn("Signed in as", page.locator("body").inner_text())
            finally:
                browser.close()

    def test_notifications_button_is_removed_from_header(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                self.assertEqual(page.get_by_role("button", name="Notifications").count(), 0)
                self.assertEqual(page.get_by_role("button", name="Sign in", exact=True).count(), 1)
            finally:
                browser.close()

    def test_header_voice_preset_chip_is_removed(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                self.assertEqual(page.locator(".preset-chip").count(), 0)
                self.assertEqual(page.locator(".topbar-title").inner_text(), "Generate Speech")
            finally:
                browser.close()

    def test_startup_suggestions_are_removed(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                self.assertEqual(page.locator(".suggestion-area").count(), 0)
                self.assertEqual(page.get_by_text("Get started with", exact=True).count(), 0)
            finally:
                browser.close()

    def test_paragraph_and_pause_controls_are_removed(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                self.assertEqual(page.locator(".editor-divider").count(), 0)
            finally:
                browser.close()

    def test_long_text_editor_grows_and_removes_resize_handle(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 768})
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                editor = page.get_by_role("textbox", name="Text to convert to speech")
                editor.fill("Long Khmer TTS text " * 150)
                page.wait_for_function(
                    "() => Number.parseFloat(document.querySelector('textarea[aria-label=\"Text to convert to speech\"]').style.height) > 130"
                )
                result = editor.evaluate(
                    "(element) => ({ height: parseFloat(element.style.height), "
                    "resize: getComputedStyle(element).resize, overflowY: getComputedStyle(element).overflowY, "
                    "scrollHeight: element.scrollHeight, clientHeight: element.clientHeight })"
                )
                self.assertLessEqual(result["height"], 330)
                self.assertEqual(result["resize"], "none")
                self.assertEqual(result["overflowY"], "auto")
                self.assertGreater(result["scrollHeight"], result["clientHeight"])
            finally:
                browser.close()

    def test_clear_button_clears_editor_text(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                editor = page.get_by_role("textbox", name="Text to convert to speech")
                clear_button = page.get_by_role("button", name="Clear")
                self.assertTrue(clear_button.is_enabled())
                editor.fill("សួស្តីពិភពលោក")
                self.assertFalse(clear_button.is_disabled())
                clear_button.click()
                self.assertEqual(editor.input_value(), "")
                self.assertTrue(clear_button.is_enabled())
            finally:
                browser.close()

    def test_searchable_voice_picker_matches_project_menu_style(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.route(
                    re.compile(r"http://localhost:8000/api/voices/"),
                    lambda route: route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps([
                            {
                                "id": "mali",
                                "name": "មាលី",
                                "description": "Clear, natural Khmer voice",
                                "control_instruction": "",
                                "has_reference_audio": False,
                            },
                            {
                                "id": "vanna",
                                "name": "វណ្ណា",
                                "description": "Warm Khmer narration voice",
                                "control_instruction": "",
                                "has_reference_audio": False,
                            },
                            *[
                                {
                                    "id": f"voice-{index}",
                                    "name": name,
                                    "description": "Natural Khmer voice",
                                    "control_instruction": "",
                                    "has_reference_audio": False,
                                }
                                for index, name in enumerate(
                                    ("ស្រីពៅ", "ចន្ទា", "សុភា", "វិសាល", "រតនា", "ដារ៉ា"),
                                    start=2,
                                )
                            ],
                        ]),
                    ),
                )
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                trigger = page.get_by_role("button", name="Choose voice")
                trigger.click()
                menu = page.get_by_role("dialog", name="Choose voice")
                self.assertTrue(menu.is_visible())
                light_styles = menu.evaluate(
                    "(element) => ({ "
                    "width: parseFloat(getComputedStyle(element).width), "
                    "radius: getComputedStyle(element).borderRadius, "
                    "background: getComputedStyle(element).backgroundColor "
                    "})"
                )
                self.assertGreaterEqual(light_styles["width"], 300)
                self.assertEqual(light_styles["radius"], "22px")
                self.assertEqual(light_styles["background"], "rgb(255, 255, 255)")
                scroll_styles = menu.locator(".voice-menu-options").evaluate(
                    "(element) => ({ "
                    "scrollbarWidth: getComputedStyle(element).scrollbarWidth, "
                    "overflow: element.scrollHeight > element.clientHeight "
                    "})"
                )
                self.assertEqual(scroll_styles["scrollbarWidth"], "thin")
                self.assertTrue(scroll_styles["overflow"])
                filter_input = menu.get_by_role("searchbox", name="Filter voices")
                filter_input.fill("វណ្ណា")
                self.assertEqual(menu.get_by_role("menuitemradio").count(), 1)
                menu.get_by_role("menuitemradio", name=re.compile("វណ្ណា")).click()
                self.assertFalse(menu.is_visible())
                self.assertIn("វណ្ណា", trigger.inner_text())

                trigger.click()
                page.evaluate("() => { document.documentElement.dataset.theme = 'dark'; }")
                dark_styles = menu.evaluate(
                    "(element) => ({ "
                    "background: getComputedStyle(element).backgroundColor, "
                    "filterBackground: getComputedStyle(element.querySelector('.voice-filter')).backgroundColor "
                    "})"
                )
                self.assertEqual(dark_styles["background"], "rgb(23, 22, 27)")
                self.assertEqual(dark_styles["filterBackground"], "rgb(33, 31, 38)")
                page.keyboard.press("Escape")
                self.assertFalse(menu.is_visible())
            finally:
                browser.close()

    def test_sidebar_hides_voice_and_settings_navigation(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                self.assertEqual(
                    page.locator(".rail-links button").evaluate_all(
                        "buttons => buttons.map(button => button.getAttribute('aria-label'))"
                    ),
                    ["Generate speech", "History"],
                )
                self.assertEqual(page.locator(".rail-bottom").count(), 0)
            finally:
                browser.close()

    def test_topbar_utilities_are_next_to_account(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                actions = page.locator(".topbar-actions")
                utilities = actions.locator(".topbar-utilities")
                self.assertTrue(utilities.get_by_role("button", name="Switch to dark mode").is_visible())
                self.assertTrue(utilities.get_by_role("link", name="VoxCPM2 project").is_visible())
                self.assertTrue(actions.get_by_role("button", name="Sign in").is_visible())
                self.assertEqual(page.locator(".rail .theme-toggle").count(), 0)
                self.assertEqual(utilities.evaluate("(element) => getComputedStyle(element).borderRadius"), "20px")
            finally:
                browser.close()

    def test_topbar_theme_toggle_and_persistence(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                dark_toggle = page.get_by_role("button", name="Switch to dark mode")
                dark_toggle.click()
                page.wait_for_function(
                    "() => document.documentElement.dataset.theme === 'dark'"
                )
                dark_editor_background = page.locator(".editor-wrap").evaluate(
                    "(element) => getComputedStyle(element).backgroundColor"
                )
                self.assertEqual(dark_editor_background, "rgb(36, 34, 41)")
                dark_button_background = page.locator(".engine-actions button").first.evaluate(
                    "(element) => getComputedStyle(element).backgroundColor"
                )
                self.assertNotEqual(dark_button_background, "rgb(255, 255, 255)")
                global_settings_colors = page.locator(".global-card").evaluate(
                    "(element) => ({ "
                    "card: getComputedStyle(element).backgroundColor, "
                    "row: getComputedStyle(element.querySelector('.setting-row')).backgroundColor, "
                    "divider: getComputedStyle(element.querySelector('.setting-row')).borderBottomColor "
                    "})"
                )
                self.assertEqual(global_settings_colors["card"], "rgb(33, 31, 39)")
                self.assertEqual(global_settings_colors["row"], "rgba(0, 0, 0, 0)")
                self.assertNotEqual(global_settings_colors["divider"], global_settings_colors["card"])

                page.reload(wait_until="networkidle")
                page.wait_for_function(
                    "() => document.documentElement.dataset.theme === 'dark'"
                )
                self.assertEqual(
                    page.evaluate("() => localStorage.getItem('kalapaktts-theme')"),
                    "dark",
                )
                page.get_by_role("button", name="Switch to light mode").click()
                page.wait_for_function(
                    "() => document.documentElement.dataset.theme === 'light'"
                )
            finally:
                browser.close()

    def test_desktop_viewport_has_no_outer_scrollbar(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 768})
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                dimensions = page.evaluate(
                    "() => ({ "
                    "pageHeight: document.documentElement.scrollHeight, "
                    "viewportHeight: document.documentElement.clientHeight, "
                    "footerBottom: document.querySelector('.editor-footer').getBoundingClientRect().bottom, "
                    "settingsHeight: document.querySelector('.settings-scroll').scrollHeight, "
                    "settingsViewportHeight: document.querySelector('.settings-scroll').clientHeight "
                    "})"
                )
                self.assertLessEqual(dimensions["pageHeight"], dimensions["viewportHeight"])
                self.assertLessEqual(dimensions["footerBottom"], dimensions["viewportHeight"])
                self.assertGreater(
                    dimensions["settingsHeight"],
                    dimensions["settingsViewportHeight"],
                    "Settings should keep its own scroll area when its content exceeds the viewport",
                )
            finally:
                browser.close()

    def test_engine_selector_defaults_to_local_server_and_marks_offline_ready(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")

                status_layout = page.locator(".engine-status").first.evaluate(
                    "(element) => ({ display: getComputedStyle(element).display, "
                    "direction: getComputedStyle(element).flexDirection })"
                )
                self.assertEqual(status_layout, {"display": "flex", "direction": "column"})

                page.wait_for_function(
                    "() => { const logo = document.querySelector('.brand-logo-black'); "
                    "return logo && logo.complete && logo.naturalWidth > 0; }"
                )
                page.get_by_role("button", name="Sign in", exact=True).click()
                page.wait_for_function(
                    "() => { const logo = document.querySelector('.brand-logo-white'); "
                    "return logo && logo.complete && logo.naturalWidth > 0; }"
                )
                page.get_by_role("button", name="Close").click()

                engine = page.get_by_role("combobox", name="Speech engine")
                self.assertEqual(engine.get_attribute("data-value"), "server")
                engine.click()
                engine_menu = page.get_by_role("listbox", name="Speech engine")
                local_option = engine_menu.locator('[data-value="local"]')
                self.assertTrue(local_option.is_disabled())
                self.assertIn("Not available yet", local_option.inner_text())
                self.assertEqual(
                    engine_menu.locator('[data-value="server"]').inner_text(),
                    "This PC (offline)",
                )
                page.keyboard.press("Escape")
                self.assertFalse(engine_menu.is_visible())

                page.wait_for_function(
                    "() => document.querySelector('.engine-offline-status')"
                    "?.textContent?.includes('Offline-ready')",
                    timeout=30000,
                )
                offline_status = page.locator(".engine-offline-status").inner_text().lower()
                self.assertIn("works without internet", offline_status)
                self.assertIn("runs locally", offline_status)

                for label in ("Quality preset", "Pace profile", "Speaking style"):
                    self.assertIn(
                        "modern-select-trigger",
                        page.get_by_role("combobox", name=label).get_attribute("class"),
                    )

                select_dropdown_option(page, "Quality preset", "fast")
                self.assertEqual(
                    page.get_by_role("combobox", name="Quality preset").get_attribute("data-value"),
                    "fast",
                )
                self.assertIn("4 steps", page.locator(".slider-card").inner_text())
                select_dropdown_option(page, "Pace profile", "relaxed")
                self.assertEqual(page.locator("#target-cps").input_value(), "12")
                select_dropdown_option(page, "Speaking style", "news_anchor")
                style_menu = page.get_by_role("listbox", name="Speaking style")
                style_menu.wait_for(state="hidden")
                page.get_by_role("combobox", name="Speaking style").click()
                self.assertTrue(style_menu.is_visible())
                menu_styles = style_menu.evaluate(
                    "(element) => ({ radius: getComputedStyle(element).borderRadius, "
                    "background: getComputedStyle(element).backgroundColor })"
                )
                self.assertNotEqual(menu_styles["radius"], "0px")
                self.assertNotEqual(menu_styles["background"], "rgba(0, 0, 0, 0)")
                page.get_by_text("Model", exact=True).click()
                self.assertFalse(style_menu.is_visible())

                select_dropdown_option(page, "Speaking style", "custom")
                page.get_by_role("combobox", name="Speaking style").click()
                popup_geometry = page.evaluate(
                    """() => {
                      const trigger = document.querySelector('#speaking-style').getBoundingClientRect();
                      const menu = document.querySelector('#speaking-style-menu');
                      const popup = menu.getBoundingClientRect();
                      return {
                        triggerLeft: trigger.left,
                        triggerWidth: trigger.width,
                        popupLeft: popup.left,
                        popupWidth: popup.width,
                        popupRight: popup.right,
                        viewportWidth: window.innerWidth,
                        scrollTop: menu.scrollTop,
                        firstOptionTop: menu.querySelector('[role="option"]').getBoundingClientRect().top,
                        popupTop: popup.top,
                      };
                    }"""
                )
                self.assertAlmostEqual(popup_geometry["popupLeft"], popup_geometry["triggerLeft"], delta=2)
                self.assertAlmostEqual(popup_geometry["popupWidth"], popup_geometry["triggerWidth"], delta=2)
                self.assertLessEqual(popup_geometry["popupRight"], popup_geometry["viewportWidth"])
                self.assertEqual(popup_geometry["scrollTop"], 0)
                self.assertGreaterEqual(popup_geometry["firstOptionTop"], popup_geometry["popupTop"])
                selected_style = page.locator("#speaking-style-menu [data-value='custom']")
                selected_pill = selected_style.locator("span")
                self.assertEqual(
                    selected_pill.evaluate("(element) => getComputedStyle(element).borderRadius"),
                    "16px",
                )
                self.assertEqual(
                    selected_pill.evaluate("(element) => getComputedStyle(element).backgroundColor"),
                    "rgb(242, 242, 243)",
                )
                page.evaluate("document.documentElement.setAttribute('data-theme', 'dark')")
                self.assertEqual(
                    selected_pill.evaluate("(element) => getComputedStyle(element).backgroundColor"),
                    "rgb(52, 49, 59)",
                )
            finally:
                browser.close()

    def test_history_navigation_shows_searchable_generation_list(self):
        history_items = [
            {
                "id": "generation-one",
                "title": "Untitled",
                "text_input": "First Khmer greeting",
                "control_instruction": "Warm Khmer narration",
                "voice_id": "voice-one",
                "audio_duration": 3.5,
                "cfg_value": 2.5,
                "inference_timesteps": 7,
                "speed": 1.0,
                "pace_mode": "custom",
                "target_cps": 15,
                "pronunciation_dictionary": True,
                "noise_reduction": False,
                "style": "natural",
                "status": "completed",
                "created_at": "2026-11-15T12:00:00Z",
                "audio_url": "/api/tts/audio/generation-one",
            },
            {
                "id": "generation-two",
                "title": "Untitled",
                "text_input": "Second school announcement",
                "control_instruction": "",
                "voice_id": "voice-one",
                "audio_duration": 5.0,
                "cfg_value": 2.0,
                "inference_timesteps": 10,
                "speed": 0.95,
                "pace_mode": "custom",
                "target_cps": 14,
                "pronunciation_dictionary": True,
                "noise_reduction": False,
                "style": "news_anchor",
                "status": "completed",
                "created_at": "2026-11-16T12:00:00Z",
                "audio_url": "/api/tts/audio/generation-two",
            },
        ]
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00" * 2205)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    accept_downloads=True,
                    permissions=["clipboard-read", "clipboard-write"],
                )
                page = context.new_page()
                page.add_init_script(
                    "localStorage.setItem('kalapaktts-theme', 'dark'); "
                    "localStorage.setItem('kalapaktts-session', JSON.stringify({"
                    "token:'browser-test-token',user:{id:'browser-test',username:'browser-test',"
                    "email:'browser-test@example.com',full_name:'Browser Test',role:'user'}}));"
                )

                def fulfill_workspace(route):
                    if route.request.method == "PATCH" and "/api/tts/history/" in route.request.url:
                        title_update = json.loads(route.request.post_data or "{}")
                        generation = next(item for item in history_items if item["id"] in route.request.url)
                        generation["title"] = title_update["title"]
                        route.fulfill(
                            status=200,
                            content_type="application/json",
                            body=json.dumps({"id": generation["id"], "title": generation["title"]}),
                        )
                    elif "/api/tts/history" in route.request.url:
                        route.fulfill(
                            status=200,
                            content_type="application/json",
                            body=json.dumps({"items": history_items, "total": len(history_items), "page": 1, "limit": 100, "pages": 1}),
                        )
                    else:
                        route.fulfill(status=200, content_type="application/json", body="[]")

                regenerate_payloads = []

                def fulfill_generation(route):
                    if route.request.url.endswith("/api/tts/status"):
                        route.fulfill(
                            status=200,
                            content_type="application/json",
                            body=json.dumps({"model_loaded": True, "model_loading": False, "download_percent": 100}),
                        )
                        return
                    regenerate_payloads.append(json.loads(route.request.post_data or "{}"))
                    route.fulfill(
                        status=200,
                        content_type="text/event-stream",
                        body=(
                            "data: "
                            + json.dumps({
                                "type": "complete",
                                "generation": {
                                    "id": "generation-retry",
                                    "audio_url": "/api/tts/audio/generation-retry",
                                    "duration": 3.5,
                                    "created_at": "2026-11-17T12:00:00Z",
                                },
                            })
                            + "\n\n"
                        ),
                    )

                page.route(re.compile(r"http://localhost:8000/api/(voices/|tts/history)"), fulfill_workspace)
                page.route(
                    re.compile(r"http://localhost:8000/api/tts/(status|generate/stream)"),
                    fulfill_generation,
                )
                page.route(
                    re.compile(r"http://localhost:8000/api/tts/audio/.*"),
                    lambda route: route.fulfill(status=200, content_type="audio/wav", body=wav_buffer.getvalue()),
                )
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")
                page.locator('.rail-button[aria-label="History"]').click()

                rows = page.locator(".history-record")
                self.assertEqual(rows.count(), 2)
                self.assertIn("2 generations", page.locator(".history-stats").inner_text())
                self.assertIn("First Khmer greeting", rows.nth(0).inner_text())
                self.assertIn("Completed", rows.nth(0).inner_text())
                dark_surfaces = page.evaluate(
                    """() => ({
                      stats: getComputedStyle(document.querySelector('.history-stat')).backgroundColor,
                      search: getComputedStyle(document.querySelector('.history-search')).backgroundColor,
                      records: getComputedStyle(document.querySelector('.history-records')).backgroundColor,
                      text: getComputedStyle(document.querySelector('.history-record-title strong')).color,
                    })"""
                )
                self.assertEqual(dark_surfaces["stats"], "rgb(41, 39, 48)")
                self.assertEqual(dark_surfaces["search"], "rgb(33, 31, 39)")
                self.assertEqual(dark_surfaces["records"], "rgb(33, 31, 39)")
                self.assertEqual(dark_surfaces["text"], "rgb(240, 238, 245)")

                search = page.get_by_role("searchbox", name="Search generations")
                search.fill("school announcement")
                self.assertEqual(rows.count(), 1)
                self.assertIn("Second school announcement", rows.first.inner_text())

                search.fill("")
                page.get_by_role("button", name="View details for generation 1").click()
                details = page.get_by_role("dialog")
                self.assertIn("First Khmer greeting", details.inner_text())
                self.assertIn("Voice", details.inner_text())
                self.assertIn("Length", details.inner_text())
                self.assertIn("Characters", details.inner_text())
                self.assertEqual(
                    details.locator(".history-detail-script").evaluate(
                        "(element) => getComputedStyle(element).backgroundColor"
                    ),
                    "rgb(33, 31, 39)",
                )
                details.get_by_role("button", name="Edit title").click()
                title_input = details.get_by_role("textbox", name="Generation title")
                title_input.fill("Khmer welcome")
                details.get_by_role("button", name="Save title").click()
                self.assertEqual(details.locator("#history-detail-title").inner_text(), "Khmer welcome")
                self.assertIn("Khmer welcome", rows.first.inner_text())
                page.keyboard.press("Escape")
                self.assertEqual(page.locator(".history-detail-modal").count(), 0)
                page.get_by_role("button", name="View details for generation 1").click()
                details = page.get_by_role("dialog", name="Khmer welcome")
                self.assertEqual(details.locator("#history-detail-title").inner_text(), "Khmer welcome")
                details.get_by_role("button", name="Copy").click()
                self.assertEqual(details.get_by_role("status").inner_text(), "Text copied.")
                details.get_by_role("button", name="Play audio").click()
                details.locator(".history-detail-player audio").wait_for()
                with page.expect_download():
                    details.get_by_role("button", name="Download audio").click()
                page.keyboard.press("Escape")
                self.assertEqual(page.locator(".history-detail-modal").count(), 0)

                page.get_by_role("button", name="Play generation 1").click()
                page.locator(".history-audio-player audio").wait_for()
                with page.expect_download():
                    page.get_by_role("button", name="Download generation 1").click()

                page.get_by_role("button", name="View details for generation 1").click()
                with page.expect_response(
                    lambda response: response.url.endswith("/api/tts/generate/stream")
                    and response.request.method == "POST"
                ):
                    page.get_by_role("button", name="Regenerate speech").click()
                page.locator(".audio-result audio").wait_for()
                regenerated = regenerate_payloads[0]
                self.assertEqual(regenerated["text"], "First Khmer greeting")
                self.assertEqual(regenerated["voice_preset_id"], "voice-one")
                self.assertEqual(regenerated["control_instruction"], "Warm Khmer narration")
                self.assertEqual(regenerated["cfg_value"], 2.5)
                self.assertEqual(regenerated["inference_timesteps"], 7)
                self.assertEqual(regenerated["speed"], 1.0)
                self.assertEqual(regenerated["target_cps"], 15)
                self.assertEqual(page.get_by_role("textbox", name="Text to convert to speech").input_value(), "First Khmer greeting")
            finally:
                browser.close()

    def test_signup_character_counter_generation_and_audio_playback(self):
        suffix = uuid.uuid4().hex[:12]
        email = f"tts-browser-{suffix}@example.com"
        username = f"tts_browser_{suffix}"
        password = f"Kalapak-{uuid.uuid4().hex}"
        fixture_text = (BACKEND_ROOT / "tests" / "fixtures" / "khmer_long_sample.txt").read_text(
            encoding="utf-8"
        ).rstrip("\r\n")
        long_sample = fixture_text.splitlines()[0]

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto("http://127.0.0.1:3000", wait_until="networkidle")

                page.get_by_role("button", name="Choose voice").click()
                vanna_option = page.get_by_role("menuitemradio", name="វណ្ណា", exact=True)
                vanna_option.wait_for(state="attached", timeout=30000)
                vanna_voice_id = vanna_option.get_attribute("data-value")
                self.assertTrue(vanna_voice_id)
                voice_names = page.get_by_role("dialog", name="Choose voice").get_by_role("menuitemradio").all_text_contents()
                self.assertGreaterEqual(len(voice_names), 5)
                page.keyboard.press("Escape")

                editor = page.get_by_role("textbox", name="Text to convert to speech")
                editor.fill(long_sample)
                self.assertEqual(
                    page.locator(".counter").inner_text(),
                    f"{len(long_sample):,} / 5,000",
                )
                generate = page.locator(".generate-button")
                self.assertTrue(generate.is_enabled())
                generate.click()

                modal = page.get_by_role("dialog")
                modal.wait_for()
                page.locator("#email").fill(email)
                page.locator("#password").fill("invalid-password")
                page.locator(".auth-submit").click()
                self.assertIn("Incorrect email or password", page.locator(".auth-error").inner_text())

                page.locator(".auth-switch button").click()
                page.locator("#username").fill(username)
                page.locator("#name").fill("Kalapak browser test")
                page.locator("#password").fill(password)
                page.locator(".auth-submit").click()
                modal.wait_for(state="hidden", timeout=30000)

                session = page.evaluate("() => JSON.parse(localStorage.getItem('kalapaktts-session'))")
                self.assertTrue(session and session.get("token"))
                select_voice(page, vanna_voice_id)
                select_dropdown_option(page, "Speaking style", "news_anchor")
                self.assertEqual(
                    page.get_by_role("combobox", name="Quality preset").get_attribute("data-value"),
                    "high_quality",
                )
                self.assertIn("10 steps", page.locator(".slider-card").inner_text())
                select_dropdown_option(page, "Quality preset", "fast")
                self.assertIn("4 steps", page.locator(".slider-card").inner_text())
                speed_slider = page.get_by_role("slider", name="Speaking speed")
                speed_slider.focus()
                speed_slider.press("Home")
                speed_slider.press("ArrowRight")
                speed_slider.press("ArrowRight")
                page.wait_for_function(
                    "() => document.querySelector('.speed-current')?.textContent === '0.80x'"
                )
                self.assertEqual(page.locator(".speed-current").inner_text(), "0.80x")
                page.wait_for_function(
                    "() => { const user = JSON.parse(localStorage.getItem('kalapaktts-session')); "
                    "const preferences = JSON.parse(localStorage.getItem(`kalapaktts-preferences-${user.user.id}`)); "
                    "return preferences.style === 'news_anchor' && preferences.speed === 0.8 "
                    "&& preferences.paceMode === 'custom' && preferences.targetCps === 14 "
                    "&& preferences.pronunciationDictionary === true "
                    "&& preferences.noiseReduction === false; }"
                )
                noise_reduction = page.get_by_role("checkbox", name="Noise reduction")
                self.assertFalse(noise_reduction.is_checked())
                noise_reduction.check()
                page.wait_for_function(
                    "() => { const user = JSON.parse(localStorage.getItem('kalapaktts-session')); "
                    "const preferences = JSON.parse(localStorage.getItem(`kalapaktts-preferences-${user.user.id}`)); "
                    "return preferences.noiseReduction === true; }"
                )
                noise_reduction.uncheck()
                page.wait_for_function(
                    "() => { const user = JSON.parse(localStorage.getItem('kalapaktts-session')); "
                    "const preferences = JSON.parse(localStorage.getItem(`kalapaktts-preferences-${user.user.id}`)); "
                    "return preferences.noiseReduction === false; }"
                )
                page.evaluate(
                    "() => { const user = JSON.parse(localStorage.getItem('kalapaktts-session')); "
                    "const key = `kalapaktts-preferences-${user.user.id}`; "
                    "const preferences = JSON.parse(localStorage.getItem(key)); "
                    "preferences.paceProfile = 'custom'; preferences.targetCps = 18; "
                    "preferences.pacePreferenceVersion = 1; "
                    "preferences.noiseReduction = true; preferences.noisePreferenceVersion = 1; "
                    "localStorage.setItem(key, JSON.stringify(preferences)); }"
                )
                page.reload(wait_until="networkidle")
                page.get_by_role("combobox", name="Speaking style").wait_for()
                page.wait_for_function(
                    "() => document.querySelector('.speed-current')?.textContent === '0.80x'"
                )
                self.assertEqual(page.get_by_role("combobox", name="Speaking style").get_attribute("data-value"), "news_anchor")
                self.assertEqual(page.locator(".speed-current").inner_text(), "0.80x")
                self.assertEqual(page.locator("#target-cps").input_value(), "14")
                self.assertEqual(page.get_by_role("combobox", name="Pace profile").get_attribute("data-value"), "natural")
                self.assertFalse(page.get_by_role("checkbox", name="Noise reduction").is_checked())
                page.wait_for_function(
                    "() => { const user = JSON.parse(localStorage.getItem('kalapaktts-session')); "
                    "const preferences = JSON.parse(localStorage.getItem(`kalapaktts-preferences-${user.user.id}`)); "
                    "return preferences.noiseReduction === false && preferences.noisePreferenceVersion === 2 "
                    "&& preferences.targetCps === 14 && preferences.paceProfile === 'natural' "
                    "&& preferences.pacePreferenceVersion === 2; }"
                )
                select_dropdown_option(page, "Quality preset", "fast")
                self.assertIn("4 steps", page.locator(".slider-card").inner_text())
                select_voice(page, vanna_voice_id)
                editor = page.get_by_role("textbox", name="Text to convert to speech")
                editor.fill(long_sample)
                self.assertTrue(page.locator(".generate-button").is_enabled())
                self.assertTrue(page.get_by_role("button", name="Preview style").is_enabled())

                with page.expect_response(
                    lambda response: response.url.endswith("/api/tts/generate/stream")
                    and response.request.method == "POST",
                    timeout=600000,
                ) as response_info:
                    page.locator(".generate-button").click()
                    page.locator(".generation-progress").wait_for(timeout=30000)

                generation_response = response_info.value
                self.assertEqual(generation_response.status, 200)
                page.wait_for_function(
                    "() => document.querySelector('.generation-progress')?.textContent?.includes('Chunk 1 of')",
                    timeout=30000,
                )
                page.locator(".streamed-audio-count").wait_for(timeout=600000)
                page.get_by_text("Your Khmer speech is ready.").wait_for(timeout=1800000)
                pace_metric = page.locator(".last-pace-metric")
                pace_metric.wait_for(timeout=30000)
                self.assertRegex(pace_metric.inner_text(), r"\d+\.\d+ chars/s")
                page.wait_for_function(
                    "() => { const player = document.querySelector('audio'); return player && player.readyState >= 2 && player.duration > 0; }",
                    timeout=30000,
                )

                status_code, _, history_body = api_request(
                    "GET",
                    "/api/tts/history?limit=20",
                    token=session["token"],
                )
                self.assertEqual(status_code, 200)
                history_items = json.loads(history_body)["items"]
                generation = next(item for item in history_items if item["text_input"] == long_sample)
                self.assertIn(
                    generation["id"],
                    {item["id"] for item in history_items},
                )
                saved_generation = generation
                self.assertAlmostEqual(saved_generation["speed"], 0.8)
                self.assertEqual(saved_generation["inference_timesteps"], 4)
                self.assertEqual(saved_generation["pace_mode"], "custom")
                self.assertAlmostEqual(saved_generation["target_cps"], 14.0)
                self.assertTrue(saved_generation["pronunciation_dictionary"])
                self.assertFalse(saved_generation["noise_reduction"])
                self.assertEqual(saved_generation["style"], "news_anchor")
                page.locator(".tabs").get_by_role("button", name="History").click()
                page.get_by_role("button", name=re.compile(re.escape(long_sample[:60]))).wait_for(timeout=30000)
                self.assertIn(
                    "0.80x",
                    page.locator(".history-item").filter(has_text=long_sample[:60]).inner_text(),
                )
                status_code, headers, wav_bytes = api_request(
                    "GET",
                    generation["audio_url"],
                    token=session["token"],
                )
                self.assertEqual(status_code, 200)
                self.assertIn("audio/wav", headers.get("Content-Type", ""))
                self.assertGreater(len(wav_bytes), 44)
                with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                    self.assertGreater(wav.getnframes(), 0)
                    self.assertGreater(wav.getframerate(), 0)

                page.evaluate(
                    "() => { const player = document.querySelector('audio'); player.muted = true; return player.play(); }"
                )
                page.wait_for_function(
                    "() => { const player = document.querySelector('audio'); return player && player.currentTime > 0; }",
                    timeout=15000,
                )

                self.assertEqual(
                    api_request(
                        "DELETE",
                        f"/api/tts/history/{generation['id']}",
                        token=session["token"],
                    )[0],
                    204,
                )
                browser.close()
        finally:
            cleanup = (
                "import sys; from tests.api_test_support import remove_test_user; "
                "remove_test_user(sys.argv[1])"
            )
            subprocess.run(
                [
                    str(BACKEND_ROOT / ".venv" / "Scripts" / "python.exe"),
                    "-c",
                    cleanup,
                    email,
                ],
                cwd=BACKEND_ROOT,
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
