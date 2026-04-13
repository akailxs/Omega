import asyncio, threading, http.server, socketserver
from playwright.async_api import async_playwright

def serve():
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", 8201), handler) as httpd:
        httpd.serve_forever()
threading.Thread(target=serve, daemon=True).start()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        page.on("console", lambda msg: print(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: print(f"*** PAGE ERROR ***: {err}"))
        await page.goto("http://localhost:8201/mind_map.html", wait_until="domcontentloaded")
        await asyncio.sleep(3)
        
        # Screenshot to see what the user sees
        await page.screenshot(path="debug_screenshot.png", full_page=False)
        
        # Check if canvas is actually painting
        pixel_check = await page.evaluate("""
            (() => {
                const c = document.getElementById('c');
                const ctx = c.getContext('2d');
                const d = ctx.getImageData(c.width/2, c.height/2, 10, 10).data;
                let nonzero = 0;
                for (let i = 0; i < d.length; i++) if (d[i] > 0) nonzero++;
                return { width: c.width, height: c.height, nonzeroPixels: nonzero, sampleRGBA: [d[0],d[1],d[2],d[3]] };
            })()
        """)
        print(f"Canvas: {pixel_check}")
        
        # Check requestAnimationFrame is running
        raf_check = await page.evaluate("""
            new Promise(resolve => {
                let count = 0;
                const start = performance.now();
                function tick() {
                    count++;
                    if (performance.now() - start > 500) {
                        resolve({ frames: count, ms: Math.round(performance.now() - start) });
                        return;
                    }
                    requestAnimationFrame(tick);
                }
                requestAnimationFrame(tick);
            })
        """)
        print(f"RAF in 500ms: {raf_check}")
        
        await browser.close()

asyncio.run(main())
