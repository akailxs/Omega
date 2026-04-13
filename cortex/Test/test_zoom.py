import asyncio, threading, http.server, socketserver
from playwright.async_api import async_playwright

def serve():
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", 8202), handler) as httpd:
        httpd.serve_forever()
threading.Thread(target=serve, daemon=True).start()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        errors = []
        page.on("pageerror", lambda err: (errors.append(str(err)), print(f"PAGE ERROR: {err}")))
        await page.goto("http://localhost:8202/mind_map.html", wait_until="domcontentloaded")
        await asyncio.sleep(3)

        # Test applyRemoteZoom exists and works
        result = await page.evaluate("""
            (() => {
                const before = CAM_T.s;
                applyRemoteZoom(1.5);
                const after = CAM_T.s;
                const hudVis = document.getElementById('zoom-hud').classList.contains('visible');
                const hudText = document.getElementById('zoom-hud').textContent;
                deactivateRemoteZoom();
                return { before, after, hudVis, hudText, errors: [] };
            })()
        """)
        print(f"Zoom test: {result}")
        print(f"Page errors: {errors}")
        await browser.close()

asyncio.run(main())
