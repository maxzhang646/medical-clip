# Paper draft

`main.tex` — MICCAI / Springer LNCS format, double-blind (author block anonymised).

## Compiling

No LaTeX locally. On [Overleaf](https://www.overleaf.com): New Project → Upload
Project, or start from the Springer LNCS template and replace `main.tex`. The
class file `llncs.cls` ships with Overleaf; for a local build install MacTeX and
fetch `llncs.cls` from Springer.

## Venue

Written as an evaluation-methodology study, which is what the work is: it reports
no new method, and its central retrieval effect is not statistically significant.
That framing suits a MICCAI workshop, a reproducibility/evaluation track, or a
journal such as MELBA. It is not a fit for the MICCAI main conference, which
expects a novel method with strong results.

## What still needs doing before submission

- A figure. Currently five tables and no figure; the prompt-sensitivity result
  (spread 0.140 → 0.091) and the gained/lost churn behind the McNemar tests both
  want one.
- Related work on evaluation is thin and uncited — the report-generation metric
  critiques (RadGraph F1, CheXbert F1, RadCliQ) should be cited properly.
- Confirm the current MICCAI page limit and abstract limit for the target year.
- The ViT-B/16 vs B/32 confound is stated in Limitations but a reviewer will
  raise it first; consider whether any partial control is possible.
