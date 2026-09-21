# Frontend Backtest tab

The GitHub connection could read CS196Illinois/FA25-Group12, but returned HTTP
403 (Resource not accessible by integration) when creating a branch. The tested
frontend change is therefore included here as a patch for that repository.
It touches only App.js navigation/routing and adds Backtest.js; it does not
replace Optimizer.js or the newer controls currently deployed on Netlify.

From the frontend repository root, after downloading this patch:

```bash
git apply --check /path/to/frontend-backtest.patch
git apply /path/to/frontend-backtest.patch
```

Deploy the backend change first. The UI frames the `/backtest` page on the host
configured by REACT_APP_API_URL. The observed production API is:

```
REACT_APP_API_URL=https://portfolio-optimizer-backend-66q4.onrender.com/api/optimize
```

Build in Project/Frontend/src/my-react-app using `npm ci` and `npm run build`.
Deploy through the existing majestic-kataifi-bac07d project only. This patch
contains no Netlify site ID or deployment configuration. If the live site's source
checkout differs from the original repository, apply the same two additive
changes there rather than deploying an old Optimizer.js from the original repo.

Validation: production build succeeds with existing unused-variable lint
warnings in App.js and Optimizer.js. No frontend deployment was performed.
