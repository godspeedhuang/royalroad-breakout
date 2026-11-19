const puppeteer = require("puppeteer-core");
(async () => {
  const browser = await puppeteer.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: false, defaultViewport: null,
    args: ["--window-size=1600,1080", "--enable-unsafe-webgpu"], userDataDir: process.argv[3] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });
  await page.goto("http://localhost:8071/atlas/umap/", { waitUntil: "networkidle0", timeout: 180000 });
  await page.waitForFunction(() => !document.body.innerText.includes("Initializing"), { timeout: 180000 }).catch(() => console.log("still initializing"));
  await new Promise(r => setTimeout(r, 15000));
  await page.screenshot({ path: process.argv[2] });
  await browser.close();
})();
