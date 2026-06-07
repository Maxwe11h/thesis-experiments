# Revision plan — supervisor feedback 2026-04-21

Companion to `supervisor_notes_2026-04-21.md`. Work through it top to bottom. Tick boxes as items are addressed. **Never delete items** — when an item is resolved, mark it `[x]` and add a one-line note of what was done so the history is auditable.

## How to resume (after `/clear` or compact)

1. Read `supervisor_notes_2026-04-21.md` (raw feedback) and this file.
2. Find the first unchecked `[ ]` in Step 3 below.
3. Open the corresponding chapter file in `thesis/chapters/` and the PDF (`~/Downloads/draft-04-2026.pdf`) side by side.
4. Make edits → tick the box → add a short "done: …" note.
5. Commit per chapter or per logical group (not per tiny nitpick).

---

## Step 1 — Cross-cutting decisions (do FIRST, before chapter edits)

These reshape the whole document; doing them first avoids rewriting chapters twice.

- [x] **RQ rewrite** (notes 3, 19). Draft one catchy overarching RQ + 3 specific sub-questions (RQ1–RQ3). Confirm with supervisor before propagating. Update ch1 intro, methodology framing, and discussion to reference RQ1/RQ2/RQ3 consistently.
  - done 2026-04-21: Overarching RQ set to *"Can trajectory-based behavioural feedback guide LLMs to design better optimisation algorithms?"* Three sub-RQs (Models/Features/Framing) added to ch1 §1.3. Propagated: ch1 §1.2 lead-in previews all three axes; ch2 §2.3 now points to RQ3; ch4 intro maps pipeline stages to RQ1/RQ2/RQ3; ch6 §6.0 tags finding with RQ3; ch7 summary reframed around the new structure. Still pending: confirm wording with supervisor before final submission.
- [~] **Repetition audit** (notes 17, 27, 31, 33). Grep `ch*.tex` for repeated phrases (e.g., "behavioural feature", "11 BLADE metrics", "32 total features"). Decide where each concept is introduced once; cut elsewhere. Expect cuts in ch2, ch3, ch4.
  - partial 2026-04-21: ch2 cleaned — removed duplicate `vanStein2025Behaviour` mention (§2.5 redundant paragraph deleted), removed old §2.6 which duplicated SAGE and behavioural framings. Ch3/ch4/ch5 audit still pending.
  - 2026-04-28: ch4 \paragraph{} tags fully removed (22 → 0). Each section now flows as plain prose with topic-sentence cues, matching the ch2 cleanup pattern. New sweeping rule: do not introduce \paragraph{} in any chapter going forward.
- [~] **Abbreviation hygiene** (notes 11, 14). First occurrence = full name + (ABBR), afterwards abbreviation only. Audit LLM, AOCC, BLADE, MA-BBOB, LLaMEA.
  - partial 2026-04-21: ch1/ch2 done — `Large Language Models` → `LLMs` on 2nd occurrence (§2.3); `Area Over the Convergence Curve (AOCC)` → `AOCC` (§2.4). Ch4 line 122 still re-expands AOCC — leave for ch4 pass.
- [ ] **Terminology consistency** (note 21). Replace "overlook" with "do not capture" and similar negatively-framed verbs with neutral phrasing across the doc.
  - still pending: ch3 line 16 retains "existing metrics overlook".

## Step 2A — Stage 4 (Phase 4) full-benchmark comparison

The draft PDF supervisor reviewed was written before Stage 4 completed. Stage 4 is now finished (4 conditions × 10 seeds on 20 MA-BBOB instances, 500 candidates per run: vanilla / neutral-behavioural / SAGE / combined; gemini-3-flash, thinking disabled). The intro and methodology summaries now mention four stages; the detailed writeups below are still pending.

- [ ] **Analyse Phase 4 results.** Aggregate AOCC per condition, compute seed-level variance, produce condition-vs-condition comparisons. Starting point: `analysis/phase4_analysis.ipynb` (already exists). Decide on significance tests alongside note 8 (bootstrap CIs or Mann–Whitney).
- [x] **Write Stage 4 methodology** — new subsection in `ch4_methodology.tex`: conditions, model config (gemini-3-flash, thinking disabled, 500 candidates, 20 instances), 10 seeds.
  - done 2026-04-28: New §sec:meth-stage4 between Stage 3 and Infrastructure. Covers: 4 conditions (vanilla / neutral / sage / combined) inline (no separate SAGE subsubsection per note 18), gemini-3-flash with thinking disabled, top-5 neutral feature subset, 20 disjoint MA-BBOB instances with greedy + local search + SA selection (CV=0.080, 24/24 functions, group shares within ±1.3%), 500 candidates × 10 seeds, bootstrap CIs flagged for ch5.
  - revised 2026-04-28: thinking-disabled rationale softened — only the variance/confound argument remains; the ``thinking adds latency without quality gains'' claim removed because the Gemini thinking interface was undergoing changes during the experiment window. A thinking-on/off contrast at scale is flagged as future work. Possible failure-rate effects are noted but not investigated here.
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
- [x] Note 4 — clarify "novel metrics": what kind? (step-size dynamics, info-theoretic, adapted population dynamics, novel metrics — spell out).
  - done 2026-04-21: Contributions paragraph now spells out three categories: "existing trajectory-based features adapted for this task", "information-theoretic inspired features that treat fitness longitudinally", and "six metrics entirely new to the optimisation literature" (ch1 line 32). Each category is cited separately.
- [x] Note 5 — cite the Behaviour Space paper: https://link.springer.com/chapter/10.1007/978-3-032-15635-8_23 (NOT the BLADE paper). Update `bibliography.bib` with proper entry.
  - done 2026-04-21: `vanStein2025Behaviour` bib entry is the PPSN XVIII Behaviour Space paper (arXiv:2507.03605). Cited at ch1 line 15 (prior work) and line 30 (contribution 1 category 1 reference).
- [x] Note 6 — "novel" → "newly designed" (reduce duplication).
  - done 2026-04-21: grep shows zero occurrences of "novel" in ch1 — the word was removed during the intro restructure. Effectively resolved.
- [x] Note 3 — propagate RQ rewrite from Step 1 here.
  - done 2026-04-21: RQ + SubQ1/2/3 block in ch1 lines 23–28.
- [x] **Extra: removed "three interrelated questions" preview paragraph** — redundant with the RQ block directly below it; ch1 now flows BLADE paragraph → RQ lead-in directly.

### ch2_background.tex (PDF page 2–3)

- [x] Note 12 — cite https://link.springer.com/article/10.1007/s11721-021-00202-9 (swarm-intelligence review). Add to `bibliography.bib`.
  - done 2026-04-21: Added `aranha2022metaphor` bib entry; cited in §2.1 paragraph about limited systematic benchmarking.
- [x] Note 13 — cite https://arxiv.org/abs/2511.16201. Add to `bibliography.bib`.
  - done 2026-04-21: Added `vanStein2025Explainable` bib entry (van Stein, Kononova, Bäck — *From Performance to Understanding*); cited at end of §2.3 as broader vision motivating the thesis direction.
- [x] Note 15 — "The Behaviour Space paper" → "A follow-up work [7] extends..".
  - done 2026-04-21: Reworded both occurrences. §2.3 now says "A follow-up work~\cite{vanStein2025Behaviour} extends LLaMEA by..."; §2.5 no longer names the paper informally and reads as prose describing the line of work.
- [ ] Note 16 — wording: "implemented in the". **Couldn't locate the phrase in any chapter — flag with supervisor for exact PDF context.**
- [x] Note 10 — define $\mathcal{S}$ and $x$ explicitly in the metaheuristic problem definition.
  - done 2026-04-21: §2.1 opening sentence now spells out: $x$ is a $d$-dimensional candidate solution, $\mathcal{S}$ is the bounded feasible search space, $\mathcal{F}: \mathcal{S} \to \mathbb{R}$ is the black-box objective.
- [x] Note 11 / 14 — abbreviation fix (see Step 1) likely lives here.
  - done 2026-04-21: In ch2 only, fixed `Large Language Models` → `LLMs` (§2.3 lead), `Area Over the Convergence Curve (AOCC)` → `AOCC` (§2.4 AOCC para, already introduced in ch1). Ch4 still re-expands AOCC at line 122 — leave for ch4 pass.
- [x] **Bonus: remove `\paragraph{}` tags across ch2.**
  - done 2026-04-21: All 16 `\paragraph{}` tags across §2.2–§2.6 converted to flowing prose with topic-sentence cues (emphasised category phrases like *algorithm selection*, *modular frameworks*, etc.). Compiles cleanly.
- [x] **Extra: §2.6 ("Structured Feedback") removed entirely.**
  - done 2026-04-21: SAGE description moved into §2.3 after the Behaviour Space follow-up mention (trimmed, no forward references to MCTS-AHD/LHNS). The proximal/distal framing and thesis-investigates language were dropped from ch2 — proximal/distal moved to the opening of ch3 (with SAGE citation). The "SubQ3 framing matters" positioning was dropped as it belongs in ch1/ch4.
- [x] **Extra: italics removed from §2.2, §2.3, and §2.5.**
  - done 2026-04-21: All `\emph{...}` wrappers on category names stripped; paragraphs now rely on topic-sentence cues.
- [x] **Extra: individual methods each have their own sentence.**
  - done 2026-04-21: EvoPrompt/APE, EoH/AEL, and MCTS-AHD/LHNS are now in separate sentences rather than joined with conjunctions.
- [x] **Extra: paragraph restructure in §2.3 — LLaMEA + Behaviour Space follow-up now one paragraph; alternative search strategies (MCTS-AHD, LHNS) in their own paragraph.**
- [x] **Extra: LLM–EC taxonomy citation softened + EC+LLM survey added.**
  - done 2026-04-21: `chauhan2025ecllmsurvey` bib entry added; §2.3 opener now reads "Building on existing taxonomies of LLM–EC integration [LLaMEA, survey]". The previous `vanStein2024LLaMEA`-only citation misrepresented the three-class split (LLaMEA paper's three classes are Prompt optimisation / LLMs as EC / Code generation, while ours is Prompt optimisation / Code generation / Complete metaheuristic generation).
- [x] **Extra: metaheuristic problem statement displayed as its own equation line in §2.1**, with symbol definitions flowing below.
- [x] **Extra: `vanStein2025Explainable` citation moved from §2.3 to §2.1 final sentence** ("automated, and increasingly explainable, approaches to algorithm design").
- [x] **Extra: Rice sentence rewritten in §2.2** — replaced awkward "asks, given..., how to learn..." with "formalises the task as learning a mapping from problem features to the best-performing algorithm, drawn from a fixed portfolio of candidates evaluated on a set of problem instances."
- [x] **Extra: last paragraph of §2.5 removed as redundant with §2.3 Behaviour Space mention.**
- [x] **Extra: "A first/second family of approaches" → "One direction / A complementary direction" in §2.5.**

### ch3_features.tex (PDF page 4 area — feature-set definitions)

- [ ] Note 7 — add one sentence: why 10 behavioural features, how selected.
- [ ] Note 20 — "newly developed" wording. (Ch3 lines 14–15 still say "entirely novel metrics" and "These novel features"; also §3.6 label `sec:features:novel`.)
- [ ] Note 21 — "do not capture" (applied globally in Step 1; verify). (Ch3 line 16 still says "existing metrics overlook".)
- [x] **Extra: proximal/distal framing added at chapter opening.**
  - done 2026-04-21: Chapter now opens with a paragraph contrasting proximal (structural) and distal (behavioural) features, citing LLaMEA-SAGE as the proven-proximal case and positioning behavioural feedback as the open question. Distal is introduced first since it's the chapter's subject; proximal is the contrast. Written without naming any specific metric so no forward references to un-introduced features.
- [x] **Extra: "This section" → "This chapter"** (adjacent paragraph, small nit).

### ch4_methodology.tex (PDF page 4, 6, 8)

- [x] Note 18 — remove the unnecessary SAGE subsubsection (already covered above).
  - done 2026-04-28: Verified no SAGE subsubsection currently exists in ch4 (likely already removed in an earlier pass). The new Stage 4 section describes SAGE inline as one of the four conditions, deliberately not as a separate subsubsection, per supervisor's intent.
- [x] Note 19 — RQ consistency (from Step 1).
  - done 2026-04-28: Each Stage section now opens with an explicit SubQ tie-in (Stage 1 → SubQ1, Stage 2 → SubQ2, Stage 3 → SubQ3); Stage 4 references the overarching RQ. Section headings renamed to "Stage N — ..." for parallel structure.
- [x] Note 22 — verify `2000 × d` vs `2000d` notation; use `2{,}000 \times d` in LaTeX for readability.
  - done 2026-04-28: `B = 2000 \times d` → `B = 2{,}000 \times d` at the evaluation-protocol paragraph; new Stage 4 section uses the same notation.
- [x] Note 23 — justify the 0.8 threshold (top-10% cutoff? reference-set threshold?). Add a sentence of justification or cite its origin.
  - done 2026-04-28 (revised): The redundancy-pruning paragraph now describes the empirical procedure — sweep $\tau \in [0.5, 1.0]$, record the surviving subset's mean Borda rank at each step, pick the elbow. This selects $\tau = 0.8$ as the threshold producing the largest single-step improvement in mean Borda rank. The sensitivity figure (presence heatmap + Borda-quality plot from `analysis/feature_selection_sensitivity.ipynb`) is **not** placed in methodology; it goes at the end of the Stage~3 results in ch5 as a lead-in to Stage~4 results, per supervisor feedback.
- [x] Note 24 — motivate the choice: "for better answering the RQ" / "because we identified gaps in the metric space".
  - done 2026-04-28: Stage 2 section now explicitly states why a feature subset is needed (prompt length / signal dilution); each stage opening also ties the choice to its SubQ.
- [x] Note 25 — replace "optimal" with "well-working" / "effective" configuration.
  - done 2026-04-28: "dual-prompt configuration identified as optimal" → "identified as effective" in §sec:meth-llamea Mutation prompts paragraph.
- [x] Note 26 — fix incorrect claim: population size does NOT influence computational cost; evaluation budget does.
  - done 2026-04-28: Selection paragraph rewritten — dropped the "computationally prohibitive" rationale, retained only the empirical reason ((1+1) elitist outperformed population-based variants in the behaviour-space study). Also added "the surviving algorithm becomes the parent for the next generation" per note 32.

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

## Post-thesis / code cleanup (not blocking submission)

- [ ] **Rename `fitness_*` → `objective_*` throughout the code.** In `BLADE/iohblade/behaviour_metrics.py` the helper is already `get_objective` and the column is `raw_y` commented as "objective function value", but the feature functions (`fitness_sample_entropy`, `fitness_permutation_entropy`, `fitness_autocorrelation`, `fitness_lempel_ziv_complexity`, `fitness_plateau_fraction`) and the docstrings that say "raw fitness" / "binarized fitness changes" still use "fitness". For consistency with the thesis prose (which uses "objective value" to avoid collision with the LLaMEA/BLADE convention that "fitness of an algorithm" = AOCC), rename these identifiers and update all callers. Touches: `behaviour_metrics.py`, any analysis notebooks referencing these functions by name, any JSON/CSV feature-column outputs, and any feedback-string templates that report feature names to the LLM. Non-trivial because it changes serialisation keys — do as a standalone, reviewed PR after submission.

---

## Notes & decisions log

Append dated entries here as decisions get made (especially on RQ wording, significance test choice, and Note 9's quantification approach).

- 2026-04-21: Plan created from supervisor's 37 PDF annotations.
- 2026-04-21: Session progress — all ch1 supervisor notes addressed (1, 2, 3, 4, 5, 6) plus a restructure (removed the "three interrelated questions" preview paragraph, now redundant with the RQ block). Ch2 supervisor notes addressed (10, 11, 12, 13, 14, 15, 17); note 16 flagged as unlocatable. Ch2 also underwent structural cleanup: §2.6 merged into §2.3 (SAGE trimmed), italics removed, paragraph tags removed, methods split into one-per-sentence, LHNS verified as LLM-driven (not Local). Ch3 got a new proximal/distal opening paragraph to host the conceptual framing dropped from §2.6. Remaining ch3 notes (7, 20, 21) and all of ch4–ch7 still pending, plus Phase 4 writeups.
