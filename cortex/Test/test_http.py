import asyncio
from playwright.async_api import async_playwright
import threading, http.server, socketserver

def serve():
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", 8134), handler) as httpd:
        httpd.serve_forever()

threading.Thread(target=serve, daemon=True).start()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        page.on("console", lambda msg: print(f"CONSOLE [{msg.type}]: {msg.text}"))
        page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))
        await page.goto("http://localhost:8134/mind_map.html")
        await asyncio.sleep(2)
        # Siumulate mouse move to trigger hovered
        await page.mouse.move(200, 200)
        await asyncio.sleep(1)
        await page.mouse.move(500, 500)
        await asyncio.sleep(1)
        await browser.close()

asyncio.run(main())
