import asyncio, threading, http.server, socketserver
from playwright.async_api import async_playwright

def serve():
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", 8199), handler) as httpd:
        httpd.serve_forever()
threading.Thread(target=serve, daemon=True).start()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        errors = []
        page.on("console", lambda msg: print(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: (errors.append(str(err)), print(f"*** PAGE ERROR ***: {err}")))
        await page.goto("http://localhost:8199/mind_map.html", wait_until="domcontentloaded")
        await asyncio.sleep(4)
        # Check if animation loop is running
        running = await page.evaluate("typeof loop === 'function' ? 'loop exists' : 'NO loop'")
        ready = await page.evaluate("typeof ready !== 'undefined' ? ready : 'undefined'")
        nodes_len = await page.evaluate("typeof nodes !== 'undefined' ? nodes.length : -1")
        print(f"\n=== DIAGNOSTICS ===")
        print(f"loop function: {running}")
        print(f"ready: {ready}")
        print(f"nodes.length: {nodes_len}")
        print(f"Errors captured: {len(errors)}")
        for e in errors:
            print(f"  -> {e}")
        await browser.close()

asyncio.run(main())
