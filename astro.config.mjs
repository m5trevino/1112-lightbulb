import { defineConfig } from 'astro/config';

export default defineConfig({
  vite: {
    server: {
      watch: {
        ignored: ['**/public/hits/**', '**/node_modules/**'],
        followSymlinks: false
      }
    }
  }
});
