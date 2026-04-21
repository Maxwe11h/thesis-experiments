# Supervisor notes on draft-04-2026.pdf

Extracted 2026-04-21 from PDF annotations (37 notes + highlights).

## Page 1 — Introduction
1. **Highlight:** Would use `\subsubsection*{}` here, no need to have a number for 1 sentence.
2. **Highlight:** Same here, I would not make this a section.
3. **Highlight (RQ):** Make this one more catchy (short and readable) research question with a few sub questions. Overarching RQ can be loosely defined, sub-questions specific and well defined. Name them RQ1–3 so you can refer back.
4. **Highlight:** "novel metrics" — what kind of metrics?
5. **Highlight:** Refer to the behaviour paper where they were introduced (it is not the BLADE paper): https://link.springer.com/chapter/10.1007/978-3-032-15635-8_23
6. **Highlight:** A bit double — maybe "newly designed".

## Page 2 — Contributions / Background
7. Add one sentence on why 10 features and how you selected them.
8. "significantly?" — did you do significance testing / bootstrapping intervals?
9. Would be better if we can quantify: e.g. "only in X% of cases the LLM moved behaviour in the indicated direction".
10. What is S? and x — be explicit and complete.
11. Second occurrence — use full name only on first occurrence, then abbreviation.
12. Cite: https://link.springer.com/article/10.1007/s11721-021-00202-9
13. Cite: https://arxiv.org/abs/2511.16201
14. Use "LLM".

## Page 3 — Related Work
15. "The Behaviour Space paper" is informal — change to "A follow-up work [7] extends..".
16. "implemented in the" (wording fix).
17. 3rd or 4th time you mention this — reduce repetitions.

## Page 4 — Methodology
18. No need for the new subsubsection, you already talk about SAGE above.
19. This question is different from the one in the introduction — keep consistent.
20. "newly developed".
21. "do not capture" (not "overlook" — too negative).
22. `2,000 × d`?
23. Why 0.8?

## Page 6
24. Add motivation: "for better answering the research question" or "because we identified gaps in the metric space" etc.

## Page 8
25. Don't say "optimal" — it's a well-working configuration, we can never prove optimality.
26. Not quite true — population size doesn't influence computational cost; eval budget does.

## Page 9
27. **StrikeOut:** no need to repeat.
28. BBOB functions aren't that distinct from each other — some are very similar, creating disbalance (bias toward majority kinds). Note for future research.
29. "the selected".
30. "The evaluation procedure".
31. You've said this at least once, probably twice already.

## Page 10
32. "And is used for the selection of the new population (parent)!"
33. Already noted earlier.

## Page 12
34. Use `sem` instead of std — tighter bands, more readable (sem = 95% confidence bounds).

## Page 16
35. Needs more clarification: why only 5 seeds? Resource constraint? What effect size would have been detectable?

## Page 17
36. `x_spread_early`: directional feedback steers the feature in the right direction (1.55 → 0.55, past top-10% reference) yet achieves *lower* AOCC (0.793 vs neutral 0.844). Deserves its own sub-finding: **correct behavioural steering does not always imply performance improvement.**

## Page 19
37. The implication for the overall experiment design isn't drawn: directional advice for `longest_no_improvement_streak` was derived from a population including 9 weaker models, so it may have been pointing in the wrong direction for Gemini Flash specifically. Make this clearer in the text.
