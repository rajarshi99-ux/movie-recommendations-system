import asyncio
import os
# pyrefly: ignore [missing-import]
from playwright.async_api import async_playwright

async def main():
    os.makedirs('screenshots', exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 1280, 'height': 800})

        print("Navigating to frontend...")
        await page.goto('http://localhost:8080', wait_until='networkidle')
        await asyncio.sleep(2)
        
        # Take home page screenshot
        print("Capturing Home...")
        await page.screenshot(path='screenshots/home.png')
        
        try:
            # Type into search input to trigger autocomplete
            print("Capturing Autocomplete...")
            await page.fill('#search-input', 'Av')
            await asyncio.sleep(1)
            # wait for dropdown
            await page.wait_for_selector('.autocomplete', state='visible')
            await asyncio.sleep(1)
            # screenshot of autocomplete
            await page.screenshot(path='screenshots/autocomplete.png')
            
            # Press enter to trigger search
            print("Capturing Recommendations...")
            await page.keyboard.press('Enter')
            await asyncio.sleep(5) # Wait for posters to load and animations automatically
            
            # Scroll down to recommendations
            await page.mouse.wheel(0, 500)
            await asyncio.sleep(1)
            await page.screenshot(path='screenshots/recommendation.png')
            
            # Go to Explore Page
            print("Capturing Explore page...")
            await page.click('#nav-explore')
            await asyncio.sleep(3) # Wait for page switch and images
            await page.screenshot(path='screenshots/explore.png')

            # Go to About Page
            print("Capturing About page...")
            await page.click('#nav-about')
            await asyncio.sleep(2) # Wait for page switch
            await page.screenshot(path='screenshots/about.png')

        except Exception as e:
            print('Could not interact:', e)

        await browser.close()
        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
