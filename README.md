# GTA VI Trophy Rarity Forecast — ensemble forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vkaroo/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vkaroo/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PS5, five quantiles
(5/25/50/75/95%) of its earn rate on Exophase at a case's target, scored by
mean pinball loss on the logit scale.

## How to run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory and writes `<id>.out` beside it.
Node 22, standard library only, no network. Deterministic: two runs on the
same case write identical bytes. Runs in well under a second per case.

## Design: ensemble of three ingredients, blended in logit space

1. **Per-trophy curve.** For each trophy, a least-squares line of
   `logit(rate)` against `ln(1 + hours since release)`, fitted over the
   case's history and read at the target. Trophy adoption after a launch is
   roughly linear on that axis: fast early gains, slow later ones. With
   fewer than three usable reads, or no movement in the history, the line is
   flat and this ingredient is just persistence.
2. **Per-group model.** Trophies are classified from their own name and
   description (rule-based — see below) into `story`, `activities`,
   `collectibles` or `completion`. Each group's mean logit is fitted the same
   way, and a trophy's forecast is its group's curve plus the trophy's own
   offset from its group at the last read. Pooling shrinks the noise of any
   single trophy's thin counts.
3. **Refresh prior, flat history only.** Exophase refreshes a game's counts
   on its own schedule, hours apart, not continuously: a flat page can sit
   unchanged for hours and then jump. When the history shows no movement at
   all, a refresh is due, and a refresh dilutes rates — newly tracked players
   hold fewer trophies than the early syncers. The prior shifts every median
   down by a group-tilted amount and widens the spread, because the
   refresh's size is the main uncertainty. When the history does move, the
   curves in (1) and (2) already carry the refreshes that happened, so no
   extra shift is applied.

Blending (logit space): when the history moves, the median is
`0.15 * persistence + 0.50 * per-trophy curve + 0.35 * group model`. When the
history is flat, the median is persistence plus the group-tilted refresh
shift. The spread combines a random-walk term (0.5 logit units per
square-root day), the curve fit's residual, a group dispersion term, and —
for a flat history — a wide refresh term.

A trophy sitting at 100% in the last read is forecast near 100% with a
modest spread: on launch night the one 100% rate was the only rate a refresh
left untouched.

## The classifier (rule-based, no id table)

The program must work for any trophy list the case hands it, so groups are
derived from each trophy's own fields, never from a hardcoded id→group
table:

- **completion**: the platinum (`tier === 'platinum'`) or a description
  mentioning `Congratulations` / `100% Completion`.
- **story**: `Complete Chapter N`, `Complete the final mission.`, or
  `Complete <Proper Name ...>` — a named mission, where every word starts
  with an uppercase letter and none holds a digit. That admits
  `Complete Breakfast Run.` and `Complete VCIA Takedown.` while rejecting
  `Complete 250 Stunts.`, `Complete an arcade game.` and
  `Complete the final mission for Andy.`
- **collectibles**: descriptions matching meditating, hunting a unique
  animal, collectible card sets, WAiNK, discovering vehicles, exploring all
  of Leonida, earning/spending money, buying properties, registering cars.
- **activities**: everything else (races, stunts, heists, side strands,
  feats).

Validated against the Leonida Legend trophy guide's grouping of the same 55
trophies (story 12 / activities 31 / collectibles 10 / completion 2):
**0 mismatches out of 55**.

## Calibration: what launch night's refresh actually did

The kit ships no practice answers, so the refresh was reconstructed from the
facts the rules state: players 638 → 3,868 and trophies earned 943 → 4,276
between the 04:00Z and 04:30Z reads. The naive dilution,
(4276/943)/(3868/638) ≈ 0.75, is wrong: the one trophy at 100% alone
absorbs 3,230 of the 3,333 new earners, so the other 54 trophies together
gain only ~103 earners and their rates fall to ~22% of their old values on a
rate-weighted mean (a trophy gaining no new earners falls to 638/3,868
= 0.165 of its rate). That reconstruction reproduces the house baseline's
stated practice numbers (loss ≈ 1.51, typical miss ≈ 0.68pp, coverage 0),
which the naive 0.75 factor does not (it gives ≈ 0.20).

Proxy practice answers were built from that reconstruction — uniform ×0.22,
story-heavy (story ×0.38, rest ×0.165), mixed (story ×0.30, activities
×0.25, rest ×0.165), noisy (uniform + lognormal jitter σ=0.3), and a
no-refresh counterfactual — each rescaled so the non-100 rates sum to the
known refreshed total (10.54). Mean pinball loss over the two practice
cases, per scenario:

| model | uniform | story-heavy | mixed | noisy | no-refresh | mean |
|---|---|---|---|---|---|---|
| house baseline | 1.40 | 1.52 | 1.44 | 1.45 | 0.02 | 1.17 |
| curve-only (uniform shift) | 0.21 | 0.28 | 0.23 | 0.26 | 0.90 | 0.37 |
| **ensemble (this entry)** | 0.21 | 0.22 | 0.20 | 0.26 | 0.98 | 0.37 |

The ensemble is kept over the curve-only fallback because it wins on the
two mechanistically most plausible scenarios (story-heavy, mixed — new
players earn early story trophies first, so dilution is grouped, not
uniform) and on the mean across all five scenarios (0.3717 vs 0.3747),
while staying within 0.01 of curve-only on the uniform/noisy ones.
The group model's other half — pooling when the history moves — cannot be
tested on the flat practice history; it is structural (variance reduction),
not tuned. Spread σ_flat was checked at 0.55 / 0.70 / 0.85 and set to 0.70,
the middle value: the mean loss barely moves across that range, and 0.70
keeps the no-refresh counterfactual from degrading as fast as 0.55 does.

## Files

- `main.ts` — the forecaster (the entrypoint; `node /bot/main.ts`).
- `LICENSE` — MIT.
