# Sync Log — 10-minute GitHub loop

Both Claudes run a ~10-minute loop. The repo is the only channel (no SSH). Aman keeps both terminals open via AnyDesk; pushes/pulls are automatic inside each loop.

## The contract
- **Box Claude** (every ~10 min): `git pull` → do the next task in `INSTRUCTIONS_FOR_BOX.md` → append to `RESULTS_FROM_BOX.md` (end with `=== MESSAGE TO LAPTOP CLAUDE ===`) → `git add -A && git commit && git push`.
- **Laptop Claude** (every 10 min): `git pull` → read new box results → reply in `INSTRUCTIONS_FOR_BOX.md` (unblock / assign next) → advance C2 + paper → `git commit && git push`.
- Each side appends ONE line here per tick: `- [side] <commit-ish>: <what happened>`.

## Rules
- Never block: if waiting on a long task (download/training), push a status line and keep the loop alive.
- Keep secrets/data/weights out (public repo).
- If the other side is silent 2 ticks, restate your current status + open question so we don't deadlock.

---
- [laptop] init: loop armed (10 min). Waiting on box to push T1–T5 results + workspace-fix outcome.
