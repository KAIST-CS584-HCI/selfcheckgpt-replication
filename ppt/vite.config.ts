import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// One route, localhost preview. JSON edits to deck.json/theme.json trigger HMR.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, open: false },
});
