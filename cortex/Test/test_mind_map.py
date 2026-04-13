import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        page.on("console", lambda msg: print(f"CONSOLE [{msg.type}]: {msg.text}"))
        page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))
        
        await page.goto("file:///Users/akai/Library/Mobile Documents/iCloud~md~obsidian/Documents/Omega/mind_map.html")
        await asyncio.sleep(2)  # Give it time to crash
        await browser.close()

asyncio.run(main())
