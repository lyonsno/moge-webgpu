import path from 'node:path';

// Keeps A/B tooling compatible with both the old automatic demo and explicit-run UI.
export async function uploadAndRunPhoto(page, filename) {
  const input = await page.$('#file-input');
  if (!input) throw new Error('Photo file input not found');
  await input.uploadFile(filename);
  if (await page.$('#run-button')) {
    await page.waitForFunction(name => document.getElementById('selected-name')?.textContent === name &&
      !document.getElementById('run-button').disabled, {}, path.basename(filename));
    await page.click('#run-button');
    await page.waitForFunction(() => window.__mogePhoto?.phase !== 'running', { timeout: 0 });
    const state = await page.evaluate(() => window.__mogePhoto);
    if (state?.phase !== 'complete') throw new Error(`Photo failed during ${state?.phase}: ${state?.error}`);
  } else {
    await page.waitForFunction(() => document.getElementById('output')?.classList.contains('visible'), { timeout: 0 });
  }
}

export function testingBrowserPath() {
  const p = process.env.CHROME_PATH;
  if (!p || p.includes('/Applications/Google Chrome.app/')) throw new Error('Set CHROME_PATH to independent Chrome for Testing or Playwright Chromium');
  return p;
}
