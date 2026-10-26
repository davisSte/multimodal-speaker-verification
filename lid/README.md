# Local Information Disclosure evaluation

This folder adds a closed-set, 1-to-N re-identification evaluation based on
[Local Information Disclosure (LID)](https://github.com/Speech-Interaction-Technology-Aalto-U/LID).
Unlike EER, this protocol asks the privacy question directly: given one intercepted trial and
an enrolment gallery of $N$ known people, how much does the attacker learn about the correct
identity?

## Metrics

The evaluator reports the LID metrics defined by the upstream project:

- **ALID:** mean local information disclosure in bits. Zero corresponds to a uniform posterior;
  positive values disclose the target identity, and negative values mislead the attacker.
- **PDR/NDR:** fraction of trials with positive/non-positive disclosure.
- **LID+/LID-:** mean disclosure conditional on positive/non-positive disclosure.
- **LID_max:** maximum observed disclosure.

It also reports familiar closed-set identification metrics to make re-identification behavior
immediately interpretable:

- rank-1 identification rate;
- rank-5 identification rate;
- mean reciprocal rank (MRR).

Rank metrics award expected credit under random tie-breaking rather than choosing a candidate
based on CSV row order.

## Required protocol

For unbiased LID reporting, use two identity-disjoint datasets:

- a **development** set used only to learn score calibration;
- a **test** set used only to report privacy results.

Each score CSV must be an exhaustive matrix: every intercepted `trial_id` must be compared with
every enrolled speaker in that split. Each trial must have exactly one correct enrolment.
Development and test should use different speakers, and every trial within one file must use the
same gallery.

```csv
enroll_spk,trial_spk,trial_id,score
speaker_1,speaker_1,trial_001,0.82
speaker_2,speaker_1,trial_001,0.13
speaker_3,speaker_1,trial_001,-0.04
speaker_1,speaker_2,trial_002,0.10
speaker_2,speaker_2,trial_002,0.76
speaker_3,speaker_2,trial_002,0.02
```

A conventional verification trial list containing only selected positive and negative pairs is
**not sufficient**. The current `scripts/eval_trials.py` output is therefore not directly usable:
you must score every trial side against the complete enrolment gallery and retain speaker and
trial identifiers.

## Run

The main project environment already contains NumPy and scikit-learn, so no extra dependencies
are needed:

```bash
uv run python lid/evaluate.py \
  --dev-scores /path/to/dev_scores.csv \
  --test-scores /path/to/test_scores.csv \
  --output-dir runs/my_system/lid
```

The command writes:

- `results.json`: aggregate LID and identification metrics plus calibration parameters;
- `local_information_disclosures.csv`: target probability and LID for every test trial;
- `calibrated_test_scores.csv`: z-scores, LLRs, and posterior probabilities for the complete
  test score matrix.

To measure how linkability grows as evidence accumulates, create a score-matrix pair for each
condition (for example, 1, 2, 5, and 10 intercepted utterances), run the evaluator for each pair,
and compare ALID and rank metrics across conditions. The score-producing model controls how
multiple utterances are combined; this evaluator measures the resulting identity disclosure.

## Method

For each trial, raw similarity scores are z-normalized across its enrolment gallery. A balanced
one-dimensional logistic regression is learned on development comparisons and applied to test
scores. The calibrated log-likelihood ratios are normalized across each gallery with softmax.
For trial $i$, target posterior $p_i$, and gallery size $N$:

$$
\mathrm{LID}_i = \log_2(Np_i).
$$

Development data is mandatory: fitting calibration on the scores used for final reporting would
leak test information and overstate the attacker.

## Attribution

This implementation follows the method and file conventions published by:

> Šterns et al., “Goodbye Equal Error Rate, Hello Local Information Disclosure: Evaluating
> Voice Anonymisation against 1-to-N Linkage Threats,” 2026.

Reference implementation: <https://github.com/Speech-Interaction-Technology-Aalto-U/LID>.
This folder is a clean project-local implementation and does not include the upstream example
data, generated results, or plotting pipeline.
