# Portfolio Optimizer frontend

This buildable React frontend preserves the newer market selection, ticker formatting,
date range/lookback controls, API error handling, and Backtest route from the deployed
`majestic-kataifi-bac07d.netlify.app` application. The application modules were recovered
from its publicly served `main.c8c06be8.js.map` on 2026-11-02; dependencies and lockfile
come from `CS196Illinois/FA25-Group12/Project/Frontend/src/my-react-app`.

The design pass adds a responsive research workspace, accessible form labels, allocation
bars, side-by-side model estimates, and explicit loading/empty/error states. The API
request contract and optimization calculations remain unchanged. Backtest styling lives
in `../static/backtest.html`, served by the backend and embedded by `Backtest.js`.

```sh
npm ci
# Local backend; defaults to http://localhost:8000/api/optimize
npm start
# Build for the existing production backend
REACT_APP_API_URL=https://portfolio-optimizer-backend-66q4.onrender.com/api/optimize npm run build
```

Deploy `build/` through the existing **majestic-kataifi-bac07d** Netlify project only.
Set the base directory to `frontend` if using this repository as the source. Preserve
SPA routing via `public/_redirects`. Deploy the backend styling change separately.
This change does not deploy either service or change the Netlify configuration.

Design guidance: [Unslop](https://github.com/heydamir/unslop) by
[Damir Sokolovsky (@heydamir)](https://x.com/heydamir), CC BY 4.0;
adapted for Portfolio Optimizer. No generated performance data is shipped.
