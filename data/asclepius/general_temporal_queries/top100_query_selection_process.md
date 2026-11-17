# How the top 100 temporal queries were selected

**Input:** `temporal_information_queries.jsonl` (826 queries)
**Output:** `temporal_information_queries_top100.jsonl` (100 queries, same format: `query`, `patient_id`, `answer`, in the original order)
**Goal:** a smaller, cheaper set of the broadest, best-answered queries for experimenting with the agent.

## What was read

- All 826 queries were read in full.
- About 160 queries have a reference answer that says some requested detail "isn't documented". For those, only the start of each answer was read, which was enough to spot the few that are actually answered properly (e.g. #437 respiratory support devices, #284 cellulitis, #415 malaria).
- For all 100 finalists, the complete reference answer was read.

## Selection criteria

1. **Common conditions only.** For each candidate, I counted how many of the 158k notes mention the query's main condition (a plain text search of the corpus). Anything under about 100 notes was dropped. Example: neutropenic enterocolitis appears in 14 notes and was dropped; hyponatremia appears in 1,037. Most of the final 100 cover conditions found in hundreds or thousands of notes, so the agent has many real patients to search and compare.
2. **Reference answers that actually answer.** About a fifth of the reference answers say a requested detail isn't documented. Those were set aside. On reading the finalists' answers, six more were dropped because they were weak in the same way: #296, #58, #231, #125, #567 (each says the key detail isn't documented) and #197 (never gives the "how soon").
3. **Genuinely about time, and natural to ask.** Kept: queries that need order, intervals, the timing of changes, or measurements over time, and that read like something a clinician would really ask, e.g. "how long did ventilation continue, and on which hospital day were they extubated". Left out: overly specific queries such as "S. sanguinegens" or "Arm 3".
4. **Varied.** No condition is repeated heavily: 4 COVID, 3 TB, 2 heart failure. The set spans cardiology, infection, kidney, critical care, cancer and blood disorders, neurology, psychiatry, GI, endocrine, obstetrics, dermatology, eye and orthopaedics.

Shortlisted queries dropped for low breadth or overlap with a query already kept: #811, #694, #2, #287, #207, #325, #821, #555, #338 (4th COVID query), #69, #333 (4th TB query), #104, #322.
Replacements added after the six weak answers were removed: #579 (autoimmune hepatitis), #610 (pulmonary embolism leading to cardiac decompensation), #784 (COVID pneumonia medications), #40 (GPA).

## One thing to know before using this set

Because these queries are broad, many notes match each one well. Using the raw query text as BM25 keywords:

| | gold patient at rank 1 | gold patient in top 20 |
|---|---|---|
| All 826 queries | 47% | 79% |
| Top 100 | 24% | 54% |

Search hasn't got worse; there are just many valid patients now, and the single gold patient is one of many. So on this set, "gold patient in the top 20" is a weak measure of success. The answer-quality criteria matter more here: whether the patients used are relevant, whether citations back each claim, and whether claims are sized to the evidence.

## The 100 selected queries

Numbers are 0-based line indices in `temporal_information_queries.jsonl`:

7, 11, 16, 36, 39, 40, 51, 57, 63, 70, 72, 88, 89, 90, 95, 97, 101, 108, 126, 128, 144, 145, 156, 171, 181, 187, 198, 200, 203, 218, 225, 229, 243, 244, 257, 261, 270, 271, 278, 288, 290, 292, 295, 306, 308, 309, 335, 340, 347, 353, 357, 369, 372, 374, 378, 384, 388, 394, 400, 429, 437, 463, 464, 466, 470, 473, 487, 491, 497, 498, 514, 515, 520, 521, 531, 534, 537, 547, 548, 550, 573, 576, 579, 599, 609, 610, 619, 625, 626, 632, 638, 665, 672, 702, 721, 729, 731, 784, 786, 793
