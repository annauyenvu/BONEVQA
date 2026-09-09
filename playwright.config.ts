import { defineConfig } from '@playwright/test';

const PYTHON = 'D:/BaoYen_work/venv/Scripts/python.exe';

export default defineConfig({
  testDir: 'app/tests',
  timeout: 300_000,
  expect: { timeout: 60_000 },
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:8000',
    viewport: { width: 1440, height: 900 },
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: `"${PYTHON}" -m uvicorn app.api:app --host 127.0.0.1 --port 8000`,
    url: 'http://127.0.0.1:8000/api/health',
    reuseExistingServer: true,
    timeout: 120_000,
    env: {
      HF_HOME: 'D:/BaoYen_work/hf_cache',
      TMP: 'D:/BaoYen_work/tmp',
      TEMP: 'D:/BaoYen_work/tmp',
      PYTHONIOENCODING: 'utf-8',
    },
  },
});
