// gta6-trophy-rarity-forecast-01 — ensemble forecaster (Node 22, standard library only).
//
// What it forecasts: for every trophy of Grand Theft Auto VI on PS5, five quantiles
// (5/25/50/75/95%) of its earn rate on Exophase at the case's target, as a percent.
//
// How it forecasts — three ingredients, blended in logit space:
//
//   1. A per-trophy curve. For each trophy, a least-squares line of logit(rate)
//      against ln(1 + hours since the game's release), fitted over the case's
//      history, read at the target. Trophy adoption after a launch is roughly
//      linear on that axis: early gains are fast, later gains slow. With fewer
//      than three usable reads, or no movement at all in the history, the line
//      is flat and this ingredient is just persistence.
//   2. A per-group model. Trophies are classified from their own name and
//      description (rule-based, no id table — see classify()) into story,
//      activities, collectibles or completion. Each group's mean logit is fitted
//      the same way as (1), and a trophy's forecast is its group's curve plus
//      the trophy's own offset from its group at the last read. Pooling shrinks
//      the noise of any single trophy's thin counts.
//   3. A refresh prior, used only when the history is perfectly flat. Exophase
//      refreshes a game's counts on its own schedule, hours apart, not
//      continuously: a flat page can sit unchanged for hours and then jump.
//      When the history shows no movement at all, a refresh is due, and a
//      refresh dilutes rates — newly tracked players hold fewer trophies than
//      the early syncers. The prior shifts every median down by a group-tilted
//      amount (early story trophies dilute least, completion most), and widens
//      the spread a lot, because the refresh's size is the main uncertainty.
//      When the history does move, the curves in (1) and (2) already carry the
//      refreshes that happened, so no extra shift is applied.
//
// Blending (logit space): when the history moves, the median is
// 0.15 * persistence + 0.50 * per-trophy curve + 0.35 * group model. When the
// history is flat, the median is persistence plus the group-tilted refresh
// shift. The spread combines a random-walk term (0.5 logit units per
// square-root day, as the house baseline), the curve fit's residual, a group
// dispersion term, and — for a flat history — a wide refresh term.
//
// Calibration (proxy practice answers reconstructed from launch night's
// refresh — see README.md): mean pinball loss over the two practice cases
// 0.3717 for this ensemble vs 0.3747 curve-only and 1.1663 house baseline,
// across five refresh scenarios (uniform, story-heavy, mixed, noisy,
// no-refresh counterfactual). SIGMA_FLAT was checked at 0.55 / 0.70 / 0.85
// and set to 0.70, the middle value.
//
// Edge cases: a trophy sitting at 100% in the last read is forecast near 100%
// with a modest spread (a refresh left the one 100% rate untouched on launch
// night). Rates are clamped to 0.01..99.99 before the logit, exactly as the
// scorer clamps them. The program is fully deterministic: no randomness, no
// clock, no network; two runs on the same case write identical bytes.
//
// Usage: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
// (defaults /input and /output, as the grader runs it).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Random-walk spread, logit units per square-root day. */
const SIGMA_RW = 0.5;
/** Extra spread for the spread between trophies of one group. */
const SIGMA_GROUP = 0.15;
/** Extra spread when the history is flat: the next refresh's size is unknown. */
const SIGMA_FLAT = 0.7;
/** Spread floor / cap, logit units. */
const SIGMA_MIN = 0.25;
const SIGMA_MAX = 1.2;

/**
 * Refresh prior, logit units, applied only when the history is perfectly flat.
 * A refresh dilutes every rate except 100%: newly tracked players hold fewer
 * trophies. Launch night shows how hard a refresh hits: players 638 -> 3,868
 * and trophies earned 943 -> 4,276, but the one trophy at 100% alone absorbs
 * 3,230 of the 3,333 new earners, so the other 54 trophies together gain only
 * ~103 earners and their rates fall to ~22% of their old values on a
 * rate-weighted mean (a trophy gaining no new earners falls to 638/3,868
 * = 0.165 of its rate). The shift below is that dilution, tilted by group:
 * new players reach the opening chapters first, so story trophies dilute
 * least; almost no newcomer has finished a collection or the whole game, so
 * collectibles and completion sit near the no-new-earners floor.
 */
const REFRESH_SHIFT: Record<string, number> = {
  story: -1.05,
  activities: -1.5,
  collectibles: -1.8,
  completion: -1.85,
};

/** Blend weights for a history that moves: persistence / curve / group. */
const W_PERSIST = 0.15;
const W_CURVE = 0.5;
const W_GROUP = 0.35;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date: string };
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => Math.log(clamp(pct) / (100 - clamp(pct)));
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/**
 * Classify a trophy into a group from its own name/description/tier only.
 * No id table: the program must work for any trophy list the case hands it.
 *
 * - completion: the platinum ("Congratulations! You're a Leonida Legend.")
 *   and 100% completion — trophies that need everything else done first.
 * - story: chapter completions and named story missions. A named mission is
 *   "Complete <Proper Name ...>" — every word capitalised, no numbers, no
 *   lowercase article — which separates "Complete Breakfast Run." from
 *   "Complete 250 Stunts." or "Complete an arcade game.".
 * - collectibles: hunting, meditating, discovering, exploring, card sets,
 *   spending/earning/laundering money, buying properties, registering cars.
 * - activities: everything else (races, stunts, heists, side strands, feats).
 */
function classify(t: Trophy): string {
  const d = t.description;
  if (t.tier === 'platinum' || /Congratulations|100% Completion/.test(d)) return 'completion';
  if (/Chapter \d/.test(d)) return 'story';
  if (/^Complete the final mission\.$/.test(d)) return 'story';
  const m = /^Complete (.+)\.?$/.exec(d);
  if (m) {
    // A named mission: every word starts with an uppercase letter and none
    // holds a digit. That admits "Complete Breakfast Run." and "Complete
    // VCIA Takedown." while rejecting "Complete 250 Stunts.", "Complete an
    // arcade game." and "Complete the final mission for Andy.".
    const words = m[1].split(/\s+/);
    if (words.length >= 1 && words.every((w) => /^[A-Z]/.test(w) && !/\d/.test(w))) return 'story';
  }
  if (/Meditat|Hunt a unique|Collectible|WAiNK|Discover \d+|Explore all of|Earn \$|Spend \$|Purchase all|Register \d+/.test(d)) return 'collectibles';
  return 'activities';
}

/** Least-squares line y = a + b*x; returns [intercept, slope]. Degenerate → flat. */
function ols(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  if (n < 2) return [ys[0] ?? 0, 0];
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 1e-12) return [my, 0];
  const b = sxy / sxx;
  return [my - b * mx, b];
}

/** Hours from the game's release (midnight UTC on release_date) to an instant. */
function hoursSinceRelease(releaseMs: number, iso: string): number {
  return Math.max(0, (Date.parse(iso) - releaseMs) / 3_600_000);
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const n = c.trophies.length;
  const last = c.history[c.history.length - 1];
  if (last === undefined) throw new Error(`${c.id} has no history`);
  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const horizonDays = c.horizon_minutes / 1440;

  // Time axis: ln(1 + hours since release) for every read and for the target.
  const xs = c.history.map((r) => Math.log1p(hoursSinceRelease(releaseMs, r.at)));
  const xTarget = Math.log1p(hoursSinceRelease(releaseMs, c.target));

  // Last non-null rate per trophy (the last read normally has them all).
  const lastRate: number[] = c.trophies.map((_, i) => {
    for (let r = c.history.length - 1; r >= 0; r--) {
      const v = c.history[r].rates[i];
      if (v !== null && v !== undefined) return v;
    }
    return 1;
  });

  // Is the history perfectly flat? Same players, same rates, every read.
  let flat = true;
  for (const r of c.history) {
    if (r.players_tracked !== c.history[0].players_tracked) flat = false;
    for (let i = 0; i < n; i++) {
      if (r.rates[i] !== null && r.rates[i] !== c.history[0].rates[i]) flat = false;
    }
  }

  const groups = c.trophies.map(classify);

  // (1) Per-trophy curve on the logit scale, plus its fit residual.
  const curveAt: number[] = new Array(n);
  const residStd: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const px: number[] = [];
    const py: number[] = [];
    for (let r = 0; r < c.history.length; r++) {
      const v = c.history[r].rates[i];
      if (v !== null && v !== undefined) {
        px.push(xs[r]);
        py.push(logit(v));
      }
    }
    if (px.length >= 3) {
      const [a, b] = ols(px, py);
      curveAt[i] = a + b * xTarget;
      let ss = 0;
      for (let k = 0; k < px.length; k++) ss += (py[k] - (a + b * px[k])) ** 2;
      residStd[i] = Math.sqrt(ss / Math.max(1, px.length - 2));
    } else {
      curveAt[i] = logit(lastRate[i]);
      residStd[i] = 0;
    }
  }

  // (2) Per-group curve: mean logit of the group's trophies per read
  // (trophies at 100% are left out — one 9.2-logit point would swamp the mean),
  // fitted the same way, read at the target.
  const groupNames = ['story', 'activities', 'collectibles', 'completion'];
  const groupAt: Record<string, number> = {};
  const groupMeanLast: Record<string, number> = {};
  for (const g of groupNames) {
    const members: number[] = [];
    for (let i = 0; i < n; i++) if (groups[i] === g) members.push(i);
    if (members.length === 0) continue;
    const gx: number[] = [];
    const gy: number[] = [];
    const lastVals: number[] = [];
    for (let r = 0; r < c.history.length; r++) {
      let sum = 0;
      let cnt = 0;
      for (const i of members) {
        const v = c.history[r].rates[i];
        if (v !== null && v !== undefined && v < 99.99) {
          sum += logit(v);
          cnt++;
        }
      }
      if (cnt > 0) {
        gx.push(xs[r]);
        gy.push(sum / cnt);
      }
      if (r === c.history.length - 1) {
        for (const i of members) {
          const v = c.history[r].rates[i];
          if (v !== null && v !== undefined && v < 99.99) lastVals.push(logit(v));
        }
      }
    }
    groupMeanLast[g] = lastVals.length > 0 ? lastVals.reduce((a, b) => a + b, 0) / lastVals.length : 0;
    if (gx.length >= 3) {
      const [a, b] = ols(gx, gy);
      groupAt[g] = a + b * xTarget;
    } else {
      groupAt[g] = groupMeanLast[g];
    }
  }

  // Pooled residual spread across trophies (median), for the spread term.
  const sortedResid = [...residStd].sort((a, b) => a - b);
  const sigmaResid = sortedResid.length > 0 ? sortedResid[Math.floor(sortedResid.length / 2)] : 0;

  const sigmaBase = Math.sqrt(
    SIGMA_RW * SIGMA_RW * horizonDays +
      sigmaResid * sigmaResid +
      SIGMA_GROUP * SIGMA_GROUP +
      (flat ? SIGMA_FLAT * SIGMA_FLAT : 0),
  );
  const sigma = Math.min(SIGMA_MAX, Math.max(SIGMA_MIN, sigmaBase));

  return c.trophies.map((t, i) => {
    const g = groups[i];
    const persist = logit(lastRate[i]);
    let median: number;
    let s = sigma;
    if (lastRate[i] >= 99.99) {
      // A trophy at 100% stayed at 100% through launch night's refresh.
      median = logit(99.9);
      s = 0.45;
    } else if (flat) {
      // (3) Refresh prior: the flat page is overdue a refresh, and a refresh
      // dilutes rates. Shift down by the group's amount.
      median = persist + (REFRESH_SHIFT[g] ?? -0.29);
    } else {
      const groupForecast = (groupAt[g] ?? groupMeanLast[g] ?? persist) + (persist - (groupMeanLast[g] ?? persist));
      median = W_PERSIST * persist + W_CURVE * curveAt[i] + W_GROUP * groupForecast;
    }
    const q: number[] = [];
    for (let j = 0; j < Z.length; j++) {
      const v = round4(clamp(invLogit(median + Z[j] * s)));
      q.push(j === 0 ? v : Math.max(v, q[j - 1]));
    }
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const rows = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
}
