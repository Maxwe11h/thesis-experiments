# Revision plan — supervisor feedback 2026-04-21

Companion to `supervisor_notes_2026-04-21.md`. Work through it top to bottom. Tick boxes as items are addressed. **Never delete items** — when an item is resolved, mark it `[x]` and add a one-line note of what was done so the history is auditable.

## How to resume (after `/clear` or compact)

1. Read `supervisor_notes_2026-04-21.md` (raw feedback) and this file.
2. Find the first unchecked `[ ]` in Step 3 below.
3. Open the corresponding chapter file in `docs/thesisLatex/chapters/` and the PDF (`~/Downloads/draft-04-2026.pdf`) side by side.
4. Make edits → tick the box → add a short "done: …" note.
5. Commit per chapter or per logical group (not per tiny nitpick).

---

## Step 1 — Cross-cutting decisions (do FIRST, before chapter edits)

These reshape the whole document; doing them first avoids rewriting chapters twice.

- [x] **RQ rewrite** (notes 3, 19). Draft one catchy overarching RQ + 3 specific sub-questions (RQ1–RQ3). Confirm with supervisor before propagating. Update ch1 intro, methodology framing, and discussion to reference RQ1/RQ2/RQ3 consistently.
  - done 2026-04-21: Overarching RQ set to *"Can trajectory-based behavioural feedback guide LLMs to design better optimisation algorithms?"* Three sub-RQs (Models/Features/Framing) added to ch1 §1.3. Propagated: ch1 §1.2 lead-in previews all three axes; ch2 §2.3 now points to RQ3; ch4 intro maps pipeline stages to RQ1/RQ2/RQ3; ch6 §6.0 tags finding with RQ3; ch7 summary reframed around the new structure. Still pending: confirm wording with supervisor before final submission.
- [ ] **Repetition audit** (notes 17, 27, 31, 33). Grep `ch*.tex` for repeated phrases (e.g., "behavioural feature", "11 BLADE metrics", "32 total features"). Decide where each concept is introduced once; cut elsewhere. Expect cuts in ch2, ch3, ch4.
- [ ] **Abbreviation hygiene** (notes 11, 14). First occurrence = full name + (ABBR), afterwards abbreviation only. Audit LLM, AOCC, BLADE, MA-BBOB, LLaMEA.
- [ ] **Terminology consistency** (note 21). Replace "overlook" with "do not capture" and similar negatively-framed verbs with neutral phrasing across the doc.

## Step 2A — Stage 4 (Phase 4) full-benchmark comparison

The draft PDF supervisor reviewed was written before Stage 4 completed. Stage 4 is now finished (4 conditions × 10 seeds on 20 MA-BBOB instances, 500 candidates per run: vanilla / neutral-behavioural / SAGE / combined; gemini-3-flash, thinking disabled). The intro and methodology summaries now mention four stages; the detailed writeups below are still pending.

- [ ] **Analyse Phase 4 results.** Aggregate AOCC per condition, compute seed-level variance, produce condition-vs-condition comparisons. Starting point: `analysis/phase4_analysis.ipynb` (already exists). Decide on significance tests alongside note 8 (bootstrap CIs or Mann–Whitney).
- [ ] **Write Stage 4 methodology** — new subsection in `ch4_methodology.tex`: conditions, model config (gemini-3-flash, thinking disabled, 500 candidates, 20 instances), 10 seeds.
- [ ] **Write Stage 4 results** — new section in `ch5_results.tex`: headline AOCC comparison, per-condition plots, any feature-level drill-downs.
- [ ] **Write Stage 4 discussion** — new section in `ch6_discussion.tex`: behavioural vs code feedback, complementarity (combined condition), interpretation relative to the overarching RQ.
- [ ] **Update ch7 conclusion** — integrate the Stage 4 finding (which condition wins, and what that says about the overarching RQ).
- [x] Update ch1 intro prose to four stages — done 2026-04-21.
- [x] Update ch4 chapter opening to four stages — done 2026-04-21.

## Step 2 — New analytical findings to integrate (before final chapter polish)

These require thinking, not just text edits. Do before touching ch5/ch6.

- [ ] **Note 36 — steering ≠ performance sub-finding.** For `x_spread_early`, directional feedback moved behaviour 1.55 → 0.55 (past top-10% reference) yet AOCC dropped (0.793 vs neutral 0.844). Promote to its own sub-finding in ch5 Results and reflect in ch6 Discussion. Candidate heading: *Correct behavioural steering does not always imply performance improvement.*
- [ ] **Note 37 — population-derived directional advice misleads specific models.** Directional advice for `longest_no_improvement_streak` came from a 10-model population including 9 weaker models; may have pointed wrong way for Gemini Flash. Draw the experiment-design implication explicitly in ch6.
- [ ] **Note 9 — quantify steering success.** Add a metric: % of cases LLM moved behaviour in the intended direction. Likely a new table or a sentence citing the number. Check if data already supports this in `results_phase4/` — if yes, compute and add; if no, decide whether to run.
- [ ] **Note 8 — significance testing.** Decide: bootstrap CIs (simplest), paired tests, or Mann-Whitney? Add to ch5 for main comparisons.

## Step 3 — Chapter-by-chapter pass

Edit each chapter in order. Cross-cutting fixes from Step 1 and analytical additions from Step 2 should already be drafted before entering this pass.

### ch1_introduction.tex (PDF page 1)

- [x] Note 1 — `\subsubsection*{}` instead of numbered subsection for 1-sentence section.
  - done 2026-04-21: went further per your decision — removed ALL \section{} headers from ch1; intro now flows as one chapter with prose transitions.
- [x] Note 2 — same, don't make it a full section.
  - done 2026-04-21: subsumed by the above.
- [ ] Note 4 — clarify "novel metrics": what kind? (step-size dynamics, info-theoretic, adapted population dynamics, novel metrics — spell out).
- [ ] Note 5 — cite the Behaviour Space paper: https://link.springer.com/chapter/10.1007/978-3-032-15635-8_23 (NOT the BLADE paper). Update `bibliography.bib` with proper entry.
- [ ] Note 6 — "novel" → "newly designed" (reduce duplication).
- [ ] Note 3 — propagate RQ rewrite from Step 1 here.

### ch2_background.tex (PDF page 2–3)

- [ ] Note 12 — cite https://link.springer.com/article/10.1007/s11721-021-00202-9 (swarm-intelligence review). Add to `bibliography.bib`.
- [ ] Note 13 — cite https://arxiv.org/abs/2511.16201. Add to `bibliography.bib`.
- [ ] Note 15 — "The Behaviour Space paper" → "A follow-up work [7] extends..".
- [ ] Note 16 — wording: "implemented in the".
- [ ] Note 11 / 14 — abbreviation fix (see Step 1) likely lives here.

### ch3_features.tex (PDF page 4 area — feature-set definitions)

- [ ] Note 7 — add one sentence: why 10 behavioural features, how selected.
- [ ] Note 20 — "newly developed" wording.
- [ ] Note 21 — "do not capture" (applied globally in Step 1; verify).

### ch4_methodology.tex (PDF page 4, 6, 8)

- [ ] Note 18 — remove the unnecessary SAGE subsubsection (already covered above).
- [ ] Note 19 — RQ consistency (from Step 1).
- [ ] Note 22 — verify `2000 × d` vs `2000d` notation; use `2{,}000 \times d` in LaTeX for readability.
- [ ] Note 23 — justify the 0.8 threshold (top-10% cutoff? reference-set threshold?). Add a sentence of justification or cite its origin.
- [ ] Note 24 — motivate the choice: "for better answering the RQ" / "because we identified gaps in the metric space".
- [ ] Note 25 — replace "optimal" with "well-working" / "effective" configuration.
- [ ] Note 26 — fix incorrect claim: population size does NOT influence computational cost; evaluation budget does.

### ch5_results.tex (PDF page 9–12, and 16–17 if Results extends there)

- [ ] Note 28 — add caveat: BBOB functions are not fully distinct from each other, some are similar, so sampling uniformly still has bias toward majority-kind functions. Flag for future work.
- [ ] Note 29 — "the selected" wording.
- [ ] Note 30 — "The evaluation procedure" wording.
- [ ] Note 32 — add: "And is used for the selection of the new population (parent)!"
- [ ] Note 34 — replace std bands with SEM (95% CI) in the relevant plot(s). Regenerate figure, re-save to `figures/`.
- [ ] Note 35 — justify 5 seeds: resource constraint? detectable effect size? Add a sentence or short paragraph.
- [ ] Note 36 — integrate the x_spread_early sub-finding drafted in Step 2.
- [ ] Note 8 / 9 — apply significance testing + steering-success-rate results from Step 2.

### ch6_discussion.tex (PDF page 17–19)

- [ ] Note 37 — integrate the population-derived-advice caveat drafted in Step 2.
- [ ] Re-read ch6 after ch5 changes; confirm discussion threads back to the (newly defined) RQ1–RQ3.

### ch7_conclusion.tex

- [ ] Re-read after everything above; update any summary claims touched by edits (especially "optimal" → "well-working", repetition removal, new sub-findings).

## Step 4 — Final pass

- [ ] Full-document read to catch remaining repetitions (note 17, 27, 31, 33 follow-up).
- [ ] `bibtool` / manual check of `bibliography.bib` — no duplicate keys, all three new refs (5, 12, 13) resolve.
- [ ] Recompile PDF, skim visually for orphan sections, broken refs, missing figures.
- [ ] Send revised draft to supervisor.

---

## Notes & decisions log

Append dated entries here as decisions get made (especially on RQ wording, significance test choice, and Note 9's quantification approach).

- 2026-04-21: Plan created from supervisor's 37 PDF annotations.
