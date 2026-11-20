# FTEC5660 Homework 4 — Python execution record

**Recorded on:** 17 November 2026  
**Source:** `solution_impl.py` and `smoke_test.py` included in this folder  
**Testing status:** Local representative smoke tests only; not the 60-question public-test benchmark.

## Command executed

```bash
python smoke_test.py
```

## Terminal output

```text
Table + metadata + 13 query parse cases: OK
Retrieval + unknown company/year abstention: OK
Disclosed/undisclosed generic-metric abstention: OK
Research report and earnings-call metadata/retrieval: OK
```

## What these results establish

The basic parsing, metadata, question parsing, retrieval and abstention branches pass the included synthetic fixtures. They **do not** establish the Homework 4 grade or end-to-end accuracy.

## Tests still required before submission

1. Download or fork the instructor's full repository at `https://github.com/hz0567/FTEC5660-HW4`.
2. Run `python apply_solution.py` there to insert the five implemented functions into the original `hw4.py`.
3. Run `python ablation.py --data-folder public_test` to generate measured ablation outcomes.
4. With your locally configured DeepSeek API key, run `python hw4.py --data-folder public_test` and then `python ablation.py --data-folder public_test --with-answers` for measured end-to-end accuracy.
5. Save the resulting `results.csv`, `ablation_results.csv`, and any terminal output you wish to keep as the final run record. Do not commit your `.env` or API key.

**Important:** The complete instructor runner and its public test corpus have not been executed in this workspace. The repository's README intentionally leaves benchmark scores unfilled until the real tests are run.