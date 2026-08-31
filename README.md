<<<<<<< HEAD
# ScreenITright — Web Prototype (Member 5's part)

This is the `web_app/` piece of ScreenITright: **Attack Discovery Dashboard**,
**Generation Studio**, **Defense Monitor**, and **Feedback Loop Visualization**,
built as an interactive Streamlit app so it runs standalone today and plugs
into your teammates' real code as soon as it lands.

## 1. What to do first

1. **Don't wait on your teammates.** The other four pillars (identify,
   generate, defend, feedback) may not exist yet. This app ships with
   `web_app/mock_data.py`, which fakes realistic output for all four, so
   your dashboard is fully clickable from minute one.
2. **Get it running locally** (steps below) and confirm all four tabs work.
3. **Agree on file names with your team now**, even before their code is
   ready: `identify/attack_database.py`, `generate/transaction_sim.py`,
   `defend/evaluator.py`, `feedback/loop_controller.py`. As long as they
   expose one function each (see "Integration points" below), your app
   will automatically start using real data instead of mock data —
   **zero changes needed on your end**.
4. Once the app runs, start on the **Solution Walkthrough document**
   (executive summary, methodology per pillar, screenshots from this app,
   deployment plan) — this repo gives you the screenshots for free.

## 2. Project structure

```
screenitright_webapp/
├── app.py                  # Entry point — run this
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Light base theme
├── web_app/
│   ├── __init__.py
│   ├── theme.py            # Shared visual identity (fonts, colors, masthead)
│   ├── mock_data.py        # Fake data generators (stand-ins for teammates' modules)
│   ├── dashboard.py        # Attack Discovery Dashboard + Generation Studio
│   ├── monitoring.py       # Defense Monitor
│   └── analytics.py        # Feedback Loop Visualization
```

This maps directly onto the repo layout from the solution doc — just drop
this `web_app/` folder (plus `app.py`, `.streamlit/`, `requirements.txt`)
into the root of the shared team repo alongside `identify/`, `generate/`,
`defend/`, `feedback/`.

## 2b. Visual identity

The whole look lives in `web_app/theme.py`, so restyling never means editing
chart-by-chart:
- `.streamlit/config.toml` sets Streamlit's own light base theme.
- `theme.masthead()` (called once in `app.py`) renders the centered brand
  header: a scanner-viewfinder logo mark, the **"ScreenITright"** wordmark
  set in a pixel font (Silkscreen, used *only* for the wordmark), a
  hand-drawn checkmark squiggle that draws itself in on load, and a live
  status pill.
- `theme.inject_css()` adds paper-toned cards with soft shadows, a
  centered animated tab underline, hover-lift on metrics/buttons, and
  Space Grotesk/Inter/IBM Plex Mono for headings/body/data respectively.
- `theme.apply_plotly_theme()` registers a matching light Plotly template.
- `theme.TIER_COLORS` keeps risk-tier colors (teal=cleared, amber=medium,
  orange=high, red=critical) the *only* place color carries meaning,
  consistent across every table and chart.

Charts are annotated, not bare: the Attack Landscape scatter has quadrant
labels (Priority Threats / Emerging / Contain / Low Priority), the Alerts
donut shows a live total in its center, and the Feedback Loop chart marks
each "attack evolves" / "model retrains" event directly on the timeline.

Want a different palette or wordmark font? Change the tokens at the top of
`theme.py` — every screen picks it up automatically.

## 3. Run it locally

```bash
# from inside screenitright_webapp/
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

It opens automatically at `http://localhost:8501`.

## 4. Set it up in VS Code

1. **Open the folder**: `File → Open Folder…` → select `screenitright_webapp/`.
2. **Install the Python extension** (Microsoft) from the Extensions panel if
   you don't have it — gives you linting, IntelliSense, and interpreter
   selection.
3. **Select the interpreter**: `Ctrl/Cmd+Shift+P` → "Python: Select
   Interpreter" → pick the `venv` you created above (or create one via the
   same command if you skipped step 3 in the terminal).
4. **Open a terminal in VS Code** (`` Ctrl+` ``) — it should already be
   activated inside your venv (you'll see `(venv)` in the prompt). Run:
   ```bash
   streamlit run app.py
   ```
5. VS Code will show a "Open in Browser" popup, or just open the printed
   `localhost:8501` link manually. The app **hot-reloads** — edit any file
   in `web_app/`, save, and Streamlit will prompt you to rerun in-browser.
6. (Optional) Add a `.vscode/launch.json` if you want to hit F5 to debug:
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Streamlit: ScreenITright",
         "type": "debugpy",
         "request": "launch",
         "module": "streamlit",
         "args": ["run", "app.py"]
       }
     ]
   }
   ```

## 5. Integration points for the other 4 members

Each of your teammates just needs to expose **one function** with this exact
name and return shape. Your app already tries to import the real thing first
and silently falls back to mock data if it's missing — so you can integrate
incrementally, one teammate at a time, with no merge conflicts in `web_app/`.

| Teammate | File they own | Function to expose | Should return |
|---|---|---|---|
| Member 1 (Identify) | `identify/attack_database.py` | `get_all_attacks()` | DataFrame like `mock_data.generate_attack_taxonomy()` |
| Member 2 (Generate) | `generate/transaction_sim.py` | `simulate_transactions(attack_type, volume, sophistication)` | DataFrame like `mock_data.generate_synthetic_transactions()` |
| Member 3 (Defend) | `defend/evaluator.py` | `get_live_scored_feed(n)` | DataFrame like `mock_data.generate_realtime_feed()` |
| Member 4 (Feedback) | `feedback/loop_controller.py` | `get_loop_history()` | DataFrame like `mock_data.generate_feedback_history()` |

Send them the exact column names used in `mock_data.py` — that's your shared
contract. As long as they match it, integration is a copy-paste of one file.

## 6. What's already interactive

- **Attack Discovery**: filter by category/severity/GenAI capability, scatter
  plot of feasibility vs. impact with quadrant labels, click into any attack
  for full detail.
- **Generation Studio**: pick an attack, drag sliders for volume and
  sophistication, generate on demand, see live histograms + geo chart,
  download the generated CSV.
- **Defense Monitor**: adjustable feed size, manual or auto-refresh (live
  polling every 2s), color-coded risk table, donut chart with live total.
- **Feedback Loop**: 30-day simulated attack-vs-defense line chart with
  annotated "attack evolves" / "model retrains" events and an auto-generated
  log below it.

## 7. Next steps for your part specifically

- [ ] Swap in real teammate functions as they land (see table above).
- [ ] Take screenshots of all 4 tabs for the Solution Walkthrough doc.
- [ ] Write the "Real-world deployment plan" section (this app is a good
      visual aid — mention it deploys as-is to Streamlit Community Cloud or
      any container host with zero code changes).
- [ ] Rehearse a 2–3 minute live demo path: Discovery → generate an attack →
      watch it show up scored in Defense Monitor → point at the Feedback
      Loop chart as the "closed loop" story beat.
=======
# phantomguard-frontend
>>>>>>> b7e947809f316df6213046faceb15a366e031626
