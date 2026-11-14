// FAA outage attribution quantile forecaster (make.fun polymarket-faa-attribution-01).
//
// Node 22, standard library only, no network, deterministic.
// Reads one case file /input/<id>.in (or $MAKEFUN_INPUT_DIR) and writes
// /output/<id>.out (or $MAKEFUN_OUTPUT_DIR) with five quantiles
// (0.05 / 0.25 / 0.5 / 0.75 / 0.95) of each market's Yes price for the
// 240 minutes after the case checkpoint.
//
// Model (adaptive-volatility persistence forecaster):
//   1. Median = persistence: the market's last price at the checkpoint.
//      (Drift extrapolations were tested and hurt: these prices jump and
//      then sit, they do not trend.)
//   2. Per-market robust volatility sigma from the interquartile range of
//      the price LEVELS over recent windows: max(IQR(last 120 min),
//      0.7 * IQR(last 240 min)) scaled by 1/1.349 and by 1/sqrt(window),
//      floored at 0.0001, capped at 0.005. When a market has not moved at
//      all, sigma falls back to 0.0002. Level IQRs beat move IQRs here:
//      most minutes are flat, so the IQR of 1-minute moves is 0, while
//      the level dispersion over 2-4 hours reflects how far the price
//      actually wanders.
//   3. Inner quantiles (0.25/0.75) are Gaussian, 0.75 * sigma * sqrt(h),
//      scaled by a term-structure factor: the empirical IQR of past
//      h-minute moves over the last 720 minutes divided by the Gaussian
//      IQR at that horizon, clipped to [0.7, 1.5].
//   4. Outer quantiles (0.05/0.95) blend the empirical quantiles of past
//      h-minute moves (last 720 minutes) with the Gaussian ones, 25%
//      empirical / 75% Gaussian, the Gaussian side widened by 1.3.
//      With fewer than 20 past h-minute moves the Gaussian form is used.
//      The narrower inner / wider outer pairing fits the move
//      distribution, which is more peaked and fatter-tailed than Gaussian.
//   5. Sibling-sum normalisation: the family of markets are mutually
//      exclusive attributions, so the medians are pulled toward summing to
//      1. The pull grows with horizon (strength 1 - exp(-h/30)), is applied
//      in full when the median sum is above 1 and at a quarter strength
//      when below 1, and is shared across markets in proportion to sigma.
//      The whole row shifts with its median.
//   6. Quantiles are clamped to [0, 1], forced non-decreasing within each
//      row, and rounded to 6 decimals.
// A market with no prices at all is forecast flat at 0.5.
//
// Practice score (6 NVIDIA cases, scorer in ~/workspace/makefun/scorer):
// loss 0.2588 vs house baseline 0.363589.

import * as fs from "node:fs";
import * as path from "node:path";

const HORIZON = 240;
const Z_INNER = 0.6744897501960817; // standard normal quantile at 0.75
const Z_OUTER = 1.6448536269514722; // standard normal quantile at 0.95
const IQR_TO_SIGMA = 1.349; // IQR of a standard normal
const SIGMA_FLOOR = 0.0001;
const SIGMA_CAP = 0.005;
const SIGMA_FALLBACK = 0.0002;
const SIGMA_WINDOW_SHORT = 120; // minutes, for the level-IQR estimate
const SIGMA_WINDOW_LONG = 240; // minutes, for the level-IQR estimate
const INNER_SCALE = 0.75; // inner quantiles are narrower than Gaussian
const OUTER_SCALE = 1.3; // outer quantiles are wider than Gaussian
const EMPIRICAL_MIN_MOVES = 20;
const EMPIRICAL_WINDOW = 720; // minutes of history used for h-minute moves
const EMPIRICAL_WEIGHT = 0.25; // weight on the empirical outer quantiles
const TERM_RATIO_LO = 0.7;
const TERM_RATIO_HI = 1.5;
const NORM_TAU = 30; // horizon scale (minutes) of the sibling-sum pull
const NORM_BELOW_ONE = 0.25; // pull strength factor when the median sum < 1
const NORM_WEIGHT_POWER = 1.0;

interface MarketInput {
  id: string;
  prices: number[];
}

interface CaseInput {
  case: string;
  markets: MarketInput[];
}

interface MarketForecast {
  id: string;
  rows: number[][]; // [horizon][5 quantiles], pre-normalisation
  sigma: number;
  hasPrices: boolean;
}

function quantile(sorted: number[], p: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n === 1) return sorted[0];
  const pos = p * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function sortedCopy(values: number[]): number[] {
  return [...values].sort((a, b) => a - b);
}

function iqr(values: number[]): number {
  if (values.length === 0) return 0;
  const s = sortedCopy(values);
  return quantile(s, 0.75) - quantile(s, 0.25);
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function round6(x: number): number {
  return Math.round(x * 1e6) / 1e6;
}

/**
 * Robust per-market volatility from the dispersion of recent price levels.
 * For a market wandering with per-minute sigma s, the levels over a window
 * of W minutes spread with IQR ~ 1.349 * s * sqrt(W), so s is recovered
 * by dividing by 1.349 * sqrt(W). The longer window is down-weighted by
 * 0.7 so the estimate tracks the recent regime.
 */
function robustSigma(prices: number[]): number {
  const short = iqr(prices.slice(-SIGMA_WINDOW_SHORT)) / IQR_TO_SIGMA / Math.sqrt(SIGMA_WINDOW_SHORT);
  const long = iqr(prices.slice(-SIGMA_WINDOW_LONG)) / IQR_TO_SIGMA / Math.sqrt(SIGMA_WINDOW_LONG);
  const raw = Math.max(short, 0.7 * long);
  if (!(raw > 0)) return SIGMA_FALLBACK; // no moves at all
  return clamp(raw, SIGMA_FLOOR, SIGMA_CAP);
}

/**
 * Past h-minute moves over the last EMPIRICAL_WINDOW minutes, sorted.
 * A move is prices[t] - prices[t - h] for t inside the window.
 */
function horizonMoves(prices: number[], h: number): number[] {
  const n = prices.length;
  const start = Math.max(0, n - EMPIRICAL_WINDOW);
  const moves: number[] = [];
  for (let t = start + h; t < n; t++) {
    moves.push(prices[t] - prices[t - h]);
  }
  return sortedCopy(moves);
}

interface HorizonStats {
  count: number;
  q05: number;
  q95: number;
  iqr: number;
}

/**
 * Per-horizon empirical distribution of h-minute moves, for h = 1..240.
 * A move is prices[t] - prices[t - h] over the last EMPIRICAL_WINDOW
 * minutes; horizons with too few moves carry zeros and count < threshold.
 */
function horizonMoveStats(prices: number[]): HorizonStats[] {
  const stats: HorizonStats[] = new Array(HORIZON);
  for (let h = 1; h <= HORIZON; h++) {
    const moves = horizonMoves(prices, h);
    if (moves.length >= EMPIRICAL_MIN_MOVES) {
      stats[h - 1] = {
        count: moves.length,
        q05: quantile(moves, 0.05),
        q95: quantile(moves, 0.95),
        iqr: quantile(moves, 0.75) - quantile(moves, 0.25),
      };
    } else {
      stats[h - 1] = { count: moves.length, q05: 0, q95: 0, iqr: 0 };
    }
  }
  return stats;
}

function forecastMarket(market: MarketInput): MarketForecast {
  const prices = market.prices;
  if (prices.length === 0) {
    // Edge case: a market with no prices forecasts flat at 0.5.
    const rows: number[][] = [];
    for (let h = 0; h < HORIZON; h++) rows.push([0.5, 0.5, 0.5, 0.5, 0.5]);
    return { id: market.id, rows, sigma: SIGMA_FALLBACK, hasPrices: false };
  }

  const n = prices.length;
  const x = prices[n - 1]; // persistence median: last price at the checkpoint
  const sigma = robustSigma(prices);

  // Precompute, per horizon, the empirical distribution of h-minute moves.
  const stats = horizonMoveStats(prices);

  const rows: number[][] = [];
  for (let h = 1; h <= HORIZON; h++) {
    const rootH = Math.sqrt(h);
    const gaussInner = Z_INNER * INNER_SCALE * sigma * rootH;
    const gaussOuter = Z_OUTER * OUTER_SCALE * sigma * rootH;
    const st = stats[h - 1];

    // Term-structure factor for the inner quantiles: empirical IQR of
    // h-minute moves relative to the Gaussian IQR at this horizon.
    let termRatio = 1;
    if (st.count >= EMPIRICAL_MIN_MOVES) {
      const gaussIqr = IQR_TO_SIGMA * sigma * rootH;
      if (gaussIqr > 0) {
        termRatio = clamp(st.iqr / gaussIqr, TERM_RATIO_LO, TERM_RATIO_HI);
      }
    }
    const inner = gaussInner * termRatio;

    // Outer quantiles: 25% empirical / 75% Gaussian blend.
    let q05 = x - gaussOuter;
    let q95 = x + gaussOuter;
    if (st.count >= EMPIRICAL_MIN_MOVES) {
      q05 = EMPIRICAL_WEIGHT * (x + st.q05) + (1 - EMPIRICAL_WEIGHT) * (x - gaussOuter);
      q95 = EMPIRICAL_WEIGHT * (x + st.q95) + (1 - EMPIRICAL_WEIGHT) * (x + gaussOuter);
    }

    rows.push([q05, x - inner, x, x + inner, q95]);
  }
  return { id: market.id, rows, sigma, hasPrices: true };
}

/**
 * Sibling-sum normalisation: pull the medians (and with them the whole
 * rows) toward summing to 1 across the family's markets, with a pull that
 * grows with horizon and is weaker when the sum is below 1.
 */
function normaliseSiblings(forecasts: MarketForecast[]): void {
  const live = forecasts.filter((f) => f.hasPrices);
  if (live.length < 2) return;
  const weights = live.map((f) => Math.pow(f.sigma, NORM_WEIGHT_POWER));
  const weightSum = weights.reduce((a, b) => a + b, 0);
  if (!(weightSum > 0)) return;

  for (let h = 1; h <= HORIZON; h++) {
    const rowIdx = h - 1;
    let sum = 0;
    for (const f of live) sum += f.rows[rowIdx][2];
    const gap = 1 - sum;
    if (gap === 0) continue;
    const strength = (1 - Math.exp(-h / NORM_TAU)) * (sum > 1 ? 1 : NORM_BELOW_ONE);
    for (let i = 0; i < live.length; i++) {
      const delta = (gap * strength * weights[i]) / weightSum;
      if (delta === 0) continue;
      const row = live[i].rows[rowIdx];
      for (let k = 0; k < 5; k++) row[k] += delta;
    }
  }
}

/** Clamp to [0, 1], enforce non-decreasing rows, round to 6 decimals. */
function finalise(rows: number[][]): number[][] {
  return rows.map((row) => {
    const out: number[] = new Array(5);
    let prev = 0;
    for (let k = 0; k < 5; k++) {
      const v = round6(clamp(row[k], 0, 1));
      out[k] = v < prev ? prev : v;
      prev = out[k];
    }
    return out;
  });
}

function forecastCase(doc: CaseInput): { id: string; q: number[][] }[] {
  const forecasts = doc.markets.map(forecastMarket);
  normaliseSiblings(forecasts);
  return forecasts.map((f) => ({ id: f.id, q: finalise(f.rows) }));
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  fs.mkdirSync(outputDir, { recursive: true });
  const files = fs
    .readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const doc = JSON.parse(
      fs.readFileSync(path.join(inputDir, file), "utf8"),
    ) as CaseInput;
    const forecasts = forecastCase(doc);
    const id = doc.case || file.replace(/\.in$/, "");
    fs.writeFileSync(
      path.join(outputDir, `${id}.out`),
      JSON.stringify({ forecasts }),
    );
  }
}

main();
