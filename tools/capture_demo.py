"""Captures screenshots of the local DeepShield demo (students' UI, unchanged) for the decks.

Needs the demo running (demo/start_demo.bat → web on :8080, API on :8765).
Uses the installed Microsoft Edge through Playwright (no browser download).
Usage: tools\\.venv\\Scripts\\python tools\\capture_demo.py
"""
import os
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "demo", "clips", "in_the_wild")
OUT = os.path.join(ROOT, "deliverables", "_work", "screens")
os.makedirs(OUT, exist_ok=True)

UPLOADS = [
    ("result_genuine_pass", "Mark_Zuckerberg_GENUINE_31559.wav"),
    ("result_deepfake_block", "Barack_Obama_DEEPFAKE_23208.wav"),
    ("result_review_uncertain", "Mark_Zuckerberg_DEEPFAKE_11316.wav"),
    ("result_review_too_short", "Queen_Elizabeth_II_DEEPFAKE_25978.wav"),
]


def shot(page, name, full=False):
    path = os.path.join(OUT, f"{name}.png")
    page.screenshot(path=path, full_page=full)
    print("saved", path)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    for scheme in ("light",):
        ctx = browser.new_context(viewport={"width": 1440, "height": 2200}, device_scale_factor=2, color_scheme=scheme)
        page = ctx.new_page()
        page.goto("http://localhost:8080/", wait_until="networkidle")
        page.wait_for_timeout(1500)
        shot(page, "landing")
        shot(page, "landing_full", full=True)

        page.goto("http://localhost:8080/#detector", wait_until="networkidle")
        page.wait_for_timeout(1500)
        shot(page, "detector_empty")

        for name, clip in UPLOADS:
            page.goto("http://localhost:8080/#detector", wait_until="networkidle")
            page.wait_for_timeout(800)
            page.locator('input[type="file"]').set_input_files(os.path.join(CLIPS, clip))
            page.get_by_text("Gateway routing").first.wait_for(timeout=120_000)
            page.wait_for_timeout(1200)
            card = page.get_by_text("Gateway routing").first.locator("xpath=ancestor::*[@data-slot='card'][last()]")
            shot(page, name)
            shot(page, name + "_full", full=True)
            try:
                card.screenshot(path=os.path.join(OUT, f"{name}_card.png"))
                print("saved card", name)
            except Exception as e:  # card selector is best-effort; the full-page shot is the fallback
                print("card capture skipped:", e)
        ctx.close()
    browser.close()
