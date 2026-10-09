import assert from 'node:assert/strict';
import puppeteer from 'puppeteer-core';

const executablePath = process.env.CHROME_PATH;
assert.ok(executablePath && !executablePath.includes('/Applications/Google Chrome.app/'), 'Use an independent testing browser');
const browser = await puppeteer.launch({ executablePath, headless: true });
try {
  const page = await browser.newPage();
  await page.goto(process.env.DEMO_URL || 'http://127.0.0.1:5198/', { waitUntil: 'networkidle0' });
  const scene = await page.$('#photo-canvas');
  assert.ok(scene, 'The first screen must contain the interactive photograph, not only an upload placeholder');
  assert.equal(await page.$eval('#sample-button', el => el.disabled), false);
  assert.equal(await page.$eval('#run-button', el => el.disabled), false);
  console.log('Photo entry contract passed');
} finally {
  await browser.close();
}
