# Orchestration Protocol — Laptop Claude ⇄ Box Claude

Two Claude Code instances, two machines, **one shared repo as the message bus.** No SSH, no live chat — communication is through files in this folder, relayed by Aman (`git push` / `git pull`).

## Roles
- **Laptop Claude (lead):** owns the plan, writes code/configs, writes `INSTRUCTIONS_FOR_BOX.md`, reads `RESULTS_FROM_BOX.md`, decides the next step, writes the paper.
- **Box Claude (executor):** runs on the A6000. Reads `INSTRUCTIONS_FOR_BOX.md` top-to-bottom, executes, writes outcomes to `RESULTS_FROM_BOX.md`.
- **Aman (courier + approver):** carries commits between machines and relays any blocker. Final decisions are his.

## The loop
1. **Laptop Claude** writes/updates `INSTRUCTIONS_FOR_BOX.md` (numbered tasks, exact commands, acceptance criteria) → Aman `git push`.
2. **Box Claude** (`git pull`) executes the next `[ ]` task. For each task it appends to `RESULTS_FROM_BOX.md`:
   - the command(s) actually run,
   - key output (last ~30 lines; full log path on the box),
   - **STATUS: DONE / BLOCKED / NEEDS-DECISION**, and metrics if any.
   It flips the task to `[x]` (done) or `[!]` (blocked) in `INSTRUCTIONS_FOR_BOX.md`.
   → Aman `git push`.
3. **Laptop Claude** (`git pull`) reads results, updates instructions, repeats.

## Rules
- **One source of truth per direction:** laptop writes INSTRUCTIONS, box writes RESULTS. Don't overwrite the other's file.
- **Append, don't clobber** RESULTS — keep history with timestamps (UTC) and the task number.
- **Never commit** weights/data/secrets (`.gitignore` enforces). Logs go on the box; paste only excerpts.
- **Blocked > guessing:** if a task fails twice, mark `BLOCKED`, write the exact error + what you tried, move to the next independent task.
- **Box Claude may fix the environment freely** (pip, conda, code edits) — that's the point of running you on the box. Commit any code fix with a clear message so laptop sees it.
