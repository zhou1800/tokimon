import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendTarget = env.TOKIMON_CHAT_UI_BACKEND ?? "http://127.0.0.1:8765";

  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": backendTarget,
        "/healthz": backendTarget,
      },
    },
  };
});
