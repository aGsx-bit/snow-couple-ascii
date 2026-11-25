# Snow Couple — Procedural ASCII Animation

Pure-code procedural animated ASCII winter scene (**Snowman & Snowgirl**) built with **NumPy** and **Tkinter** (`96×244` grid, `4×7` procedural glyph-atlas at `30 FPS`) — zero video files, images, or external assets behind the code.

Created by **Mahamad (`aGsx` / `@aGsx-bit`)**.

## Features

- **100% Procedural Graphics & Animation:** Signed-distance-field snowmen, layered snowy pine trees with wind sway, drifting aurora ribbons, twinkling stars, falling snowflakes, cratered moon, and flickering ground snow dust.
- **Smooth `30 FPS` Glyph-Atlas Renderer:** Renders a crisp `4×7` pure-NumPy ASCII bitmap grid (`976×672` px) directly to `tk.PhotoImage`.
- **Interactive Controls:**
  - `R` — Restart animation from `t = 0.0`
  - `Space` — Pause / Resume without losing elapsed timestamp
  - `A` — Jump straight to the final scene (`9.50s`)
  - `Esc` — Close window

## Quick Start

```bash
pip install -r requirements.txt
python snow_couple.py
```

## License & Attribution

Licensed under the [Attribution Open Source License](./LICENSE).  
Copyright © 2026 **Mahamad (`aGsx` / `@aGsx-bit`)**.

Anyone using, forking, sharing, or building on this project **must visibly credit Mahamad (`aGsx` / `@aGsx-bit`)** and preserve the original [LICENSE](./LICENSE) notice.
