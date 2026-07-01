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
- [laptop] tick1: no box push yet (git-auth/setup). Wrote docs/PAPER_OUTLINE.md; confirmed BOVText eval = bundled Task2_VideoTextSpotting/evaluation.py; reviewed C2 module (needs wiring not rebuild).
- [laptop] tick2: FIXED git push (gh credential helper; GUI manager was hanging). Added box git-auth fix to INSTRUCTIONS. Wrote docs/C2_WIRING.md. Box still no push.
- [box] 2026-06-29T07:55Z c-a/c-b/T3/T4/T5/T7 done; BOVText dl in progress class-by-class; smoke pending frames.
- [laptop] tick3: box live! answered Q1 (keep VOC 5462+chn_cls_list; tracker uses IDs not text; dont crash on Chinese) + Q2 (deepsolo_bov.pth only, NOT gomatching_iter30k=English). Assigned T6/T8/T9 + C2 shape-scout. BOVText=~430k frames (11 cls), disk fine.
- [box] 2026-06-29T08:15Z T8 smoke PASSED (loss=0.250, 2.77it/s, ckpt saved); T9 pending download; C2 wiring found complete.
- [laptop] tick4: smoke PASSED. Answered: keep cleaned train.json; missing keys expected; C2=LIVE-decode DeepSolo (no cache); WAIT full 430k then full run. Assigned T9(full)/T10(eval split)/T11(C2 snippet). Updated C2_WIRING.
- [box] 2026-06-29T13:55Z T9 LAUNCHED bov_partial (Cls7+Cls10, PID=865678, 3.34it/s, ETA~2.5h); T6 verified; full download still running Cls11+.
- [box] 2026-06-29T14:05Z T10 DONE (test GT exists: 480vids/32cls, convert_bovtext_test.py written); T11 DONE (live-decode patch spec in RESULTS); bov_test registered in vts.py.
- [box] 2026-06-29T14:10Z status: iter 2720/30k (5k ckpt ~14:21); Cls11 51% (Cls12-18=0, ETA 2-3h); test video 75%.
- [box] 2026-06-29T14:17Z 5k ckpt saved (total_loss=0.00047, 304MB); test video extracted; frame conv running; Cls11 81%.
- [box] 2026-06-29T14:30Z iter 6120/30k; Cls11 97% (~2min); test extraction 110k+ frames; full download ETA ~19:30 UTC.
- [box] 2026-06-29T14:45Z iter 8660/30k (29%); test extraction 21/32 cls (279k frames); Cls11 DONE on disk.
- [box] 2026-06-29T14:55Z T10 DONE: test.json written (427k imgs/2M anns); bov_test verified; iter 11380/30k (38%).
- [box] 2026-06-29T09:30Z (IST=14:59) iter 13080/30k (44%); Cls11_Movie downloading (78k frames); 15k ckpt ~09:40Z.
- [box] 2026-06-29T09:41Z (IST=15:11) 15k ckpt saved (loss=6.74e-06); iter 16040/30k; Cls11 done ✓; Cls12-19 dl resumed.
- [box] 2026-06-29T10:04Z (IST=15:34) 20k ckpt (loss=4.18e-05); Cls11 corrupted+redownloading; watcher patched; full dl ETA 14-16Z.
- [box] 2026-07-01T06:19Z RECOVERY: session died; bov_partial at 24360/30k (20k ckpt); BOVText 4/11 cls done; redownloading 7 cls; full training watcher armed.
- [box] 2026-07-01T06:48Z SOLE ORCHESTRATOR: loop stopped; Cls11 97%; full dl ETA ~08:15Z; full bov_train auto-starts then; done ~11:15Z.
- [box] 2026-07-01T07:50Z Cls11/14/15 done; Cls16 70%; ETA full dl ~08:50Z; training start ~08:50Z; done ~11:50Z.
- [box] 2026-07-01T08:40Z 9/11 cls done; Cls18 81%; training starts ~08:55Z; C2 patch committed.
- [box] 2026-07-01T08:50Z ALL CLASSES DONE ✓; bov_train LAUNCHED PID=22976 (3.33it/s, ETA 11:25Z, GPU 98%).
- [box] 2026-07-01T09:31Z iter 7340/30k; 5k ckpt saved (loss=0.00916); ETA ~11:25Z.
- [box] 2026-07-01T10:17Z iter 16480/30k (55%); 5k/10k/15k ckpts done; loss converged ~0 at 15k; ETA ~11:22Z.
- [box] 2026-07-01T11:25Z T9 DONE: model_final.pth 304MB, loss=8.95e-07; eval PID=29281 started (480 vids, ~2-3h).
- [box] 2026-07-01T11:55Z eval: 65/480 vids done, 26s/vid, GPU 96%; ETA ~15:00Z.
- [box] 2026-07-01T14:19Z eval: 104/480 (22%); pace ~97s/vid; ETA ~00:25Z Jul2 (later vids longer).
- [box] 2026-07-01T15:20Z eval: 135/480 (28%); later vids ~10min each; ETA ~01:00Z Jul2.
- [box] 2026-07-01T16:21Z eval: 170/480 (35%); 35 vids/hr; ETA ~01:10Z Jul2.
- [box] 2026-07-01T17:22Z eval: 209/480 (44%); ETA ~01:00Z Jul2.
- [box] 2026-07-01T18:23Z eval: 237/480 (49%); ETA ~01:00Z Jul2.
- [box] 2026-07-01T19:24Z eval: 274/480 (57%); ETA ~01:10Z Jul2.
