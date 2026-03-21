# 1112 King Rd Forensic Audio Analysis

Multi-model forensic triangulation using VGGish + BEATs + LAION-CLAP

## Models Used
- **VGGish** (Google 2017) - Raw acoustic salience (128-dim embeddings)
- **BEATs** (Microsoft 2023) - Deep structural anomaly detection (768-dim transformer)
- **LAION-CLAP** (LAION-AI 2023) - Zero-shot semantic matching (contrastive audio-text)

## Site Structure
- `/` - Main evidence page with 5 FOR_SURE events
- `/triangulation` - Multi-model convergence analysis

## Build
```bash
npm install
npm run build
```

Deploys to Netlify automatically on push to main.
