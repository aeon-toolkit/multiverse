"""Build HTML leaderboards from Multiverse benchmark results.

Results are stored one directory per estimator under ``results/multiverse``, with
one file per performance measure::

    results/multiverse/<estimator>/<estimator>_<metric>.csv

Each file holds one row per dataset and one column per resample, written by the
evaluation tooling with a ``Resamples:`` header. Where several resamples are
present they are averaged.

:func:`leaderboard` takes a list of datasets, a list of estimators, and a metric
name, and writes a self-contained HTML page holding a per-dataset table, average
scores and ranks, and a critical difference diagram.

Run this module directly to build a leaderboard over Multiverse-core for every
estimator with results in the repository::

    python -m multiverse.experiments.tables
"""

from __future__ import annotations

__maintainer__ = ["TonyBagnall"]
__all__ = [
    "leaderboard",
    "dataset_summary",
    "dataset_page",
    "dataset_markdown",
    "leaderboard_markdown",
    "write_markdown_table",
    "available_estimators",
    "load_metric",
    "load_missing_reasons",
]

import base64
import io
from datetime import date
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "multiverse"

#: A dataset whose best estimator gains no more than this over the baseline
#: shows little signal; one whose best reaches SATURATED_BEST is solved.
#: Both separate estimators poorly, for opposite reasons.
NO_SIGNAL_GAIN = 0.05
SATURATED_BEST = 0.99

#: Metrics where a smaller value is a better result. Anything not listed here is
#: treated as higher-is-better.
LOWER_IS_BETTER = {"logloss"}

#: Column headings for the metrics written by the evaluation tooling.
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balacc": "Balanced accuracy",
    "auroc": "AUROC",
    "f1": "F1",
    "logloss": "Log loss",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
}


def available_estimators(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    exclude=(),
) -> list[str]:
    """Return the estimators that have a results directory.

    Parameters
    ----------
    results_dir : Path or str
        Directory holding one sub-directory per estimator.
    exclude : iterable of str, default=()
        Estimator names to leave out, for example a baseline such as
        ``"Dummy"`` that would otherwise take a place in the leaderboard.

    Returns
    -------
    list of str
        Estimator names, sorted, case insensitively.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results directory not found: {results_dir}")
    excluded = set(exclude)
    return sorted(
        (
            path.name
            for path in results_dir.iterdir()
            if path.is_dir() and path.name not in excluded
        ),
        key=str.lower,
    )


def load_metric(
    estimator: str,
    metric: str,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> pd.Series:
    """Load one estimator's results for one metric.

    Parameters
    ----------
    estimator : str
        Estimator name, matching its results directory.
    metric : str
        Metric name, matching the file suffix, for example ``"accuracy"``.
    results_dir : Path or str
        Directory holding one sub-directory per estimator.

    Returns
    -------
    pd.Series
        Score per dataset, averaged over resamples where there is more than one.
        Named after the estimator.
    """
    path = Path(results_dir) / estimator / f"{estimator}_{metric}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no {metric} results for {estimator}: {path}")

    frame = pd.read_csv(path, index_col=0)
    series = frame.mean(axis=1)
    series.index = series.index.astype(str)
    series.name = estimator
    return series


def _critical_difference_png(
    scores: pd.DataFrame, lower_better: bool, alpha: float
) -> str | None:
    """Render a critical difference diagram as a base64 PNG, or None if it fails.

    A diagram needs at least two estimators, and the underlying tests can refuse
    degenerate input. A leaderboard is still useful without one, so failure here
    is reported on the page rather than raised.
    """
    if scores.shape[1] < 2:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aeon.visualisation import plot_critical_difference

    try:
        fig, _ = plot_critical_difference(
            scores.to_numpy(),
            list(scores.columns),
            lower_better=lower_better,
            alpha=alpha,
        )
    except Exception:  # noqa: BLE001 - diagram is optional, page is not
        return None

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


_SCRIPT = """
(function () {
  var table = document.getElementById("leaderboard");
  if (!table) return;
  var body = table.tBodies[0];
  var headers = table.querySelectorAll("th.sortable");

  function value(row, index, text) {
    var cell = row.cells[index];
    if (!cell) return text ? "" : 0;
    var raw = cell.textContent.trim();
    if (text) return raw.toLowerCase();
    var number = parseFloat(raw.replace(/,/g, ""));
    return isNaN(number) ? 0 : number;
  }

  function sort(header) {
    var index = parseInt(header.dataset.col, 10);
    var isText = header.dataset.type === "text";
    // repeat clicks toggle; a new column starts in its "best first" direction
    var current = header.getAttribute("aria-sort");
    var direction;
    if (current === "ascending") direction = "descending";
    else if (current === "descending") direction = "ascending";
    else direction = header.dataset.first === "asc" ? "ascending" : "descending";

    var sign = direction === "ascending" ? 1 : -1;
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (a, b) {
      var x = value(a, index, isText);
      var y = value(b, index, isText);
      if (x < y) return -sign;
      if (x > y) return sign;
      return 0;
    });
    rows.forEach(function (row, i) {
      body.appendChild(row);
      // the position column always numbers the order actually on screen
      if (row.cells[0]) row.cells[0].textContent = i + 1;
    });

    headers.forEach(function (other) { other.removeAttribute("aria-sort"); });
    header.setAttribute("aria-sort", direction);
  }

  headers.forEach(function (header) {
    header.tabIndex = 0;
    header.addEventListener("click", function () { sort(header); });
    header.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        sort(header);
      }
    });
  });
})();
"""

_STYLE = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #666666;
  --line: #e0e0e0; --head: #f5f5f5; --best: #e8f4ec; --accent: #2b6cb0;
  --stripe: #fafafa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16181c; --fg: #e8e8e8; --muted: #9aa0a6;
    --line: #30343a; --head: #212429; --best: #1e3226; --accent: #7aa7d9;
    --stripe: #1b1e23;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2.5rem 0 .75rem; }
.sub { color: var(--muted); margin: 0 0 2rem; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: .45rem .7rem; text-align: right; white-space: nowrap;
  border-bottom: 1px solid var(--line); }
th { background: var(--head); font-weight: 600; position: sticky; top: 0; }
td.pos { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums;
  width: 2.5rem; }
th.pos { text-align: right; color: var(--muted); }
th:nth-child(-n+2), td:nth-child(-n+2) { position: sticky; background: var(--bg); }
td:nth-child(2), th:nth-child(2) { text-align: left; }
th:nth-child(1), td:nth-child(1) { left: 0; }
th:nth-child(2), td:nth-child(2) { left: 2.5rem; box-shadow: 1px 0 0 var(--line); }
th:nth-child(-n+2) { background: var(--head); z-index: 1; }
tbody tr:last-child td { border-bottom: none; }
td.best { background: var(--best); font-weight: 600; }
th.grp { text-align: center; border-left: 2px solid var(--line); }
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { color: var(--accent); }
th.sortable::after { content: "↕"; opacity: .25; margin-left: .35em;
  font-size: .85em; }
th.sortable[aria-sort="ascending"]::after { content: "↑"; opacity: 1; }
th.sortable[aria-sort="descending"]::after { content: "↓"; opacity: 1; }
th.sortable:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
pre { background: var(--head); border: 1px solid var(--line); border-radius: 6px;
  padding: .8rem 1rem; overflow-x: auto; font-size: .86rem; line-height: 1.5;
  margin: 0; }
pre code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
ul.missing { margin: .5rem 0 0; padding-left: 1.2rem; }
ul.missing li { margin-bottom: .3rem; }
th.sep, td.sep { border-left: 2px solid var(--line); }
tbody td:not(:first-child) { font-variant-numeric: tabular-nums; }
tbody tr:nth-child(even) td { background: var(--stripe); }
tbody tr:hover td { background: var(--head); }
tbody tr:hover td.best { background: var(--best); }
tbody tr:hover td.best { background: var(--best); }
figure { margin: 0; }
figure img { max-width: 100%; height: auto; background: #fff; border-radius: 6px;
  padding: .5rem; }
.note { color: var(--muted); font-size: .88rem; margin-top: .6rem; }
details { margin-top: .6rem; }
summary { cursor: pointer; color: var(--accent); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; }
"""




def load_missing_reasons(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """Load the recorded reasons for missing results.

    Reads ``missing_results.csv`` from the results directory, which records why
    a particular estimator has no result for a particular dataset. Returns an
    empty frame with the expected columns if the file is absent, since the
    reasons are documentation rather than something the leaderboard needs.

    Returns
    -------
    pd.DataFrame
        Columns ``estimator``, ``dataset``, ``reason``, ``detail``.
    """
    columns = ["estimator", "dataset", "reason", "detail"]
    path = Path(results_dir) / "missing_results.csv"
    if not path.is_file():
        return pd.DataFrame(columns=columns)

    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame:
            frame[column] = ""
    return frame.fillna("")


def _missing_by_estimator(frames, estimators, common, datasets):
    """Work out which datasets each estimator is individually missing.

    A dataset is counted against an estimator when the dataset was requested,
    some estimator has a result for it, and this one does not on at least one
    metric. That distinguishes "this estimator failed here" from "nobody ran
    this dataset".
    """
    requested = list(dict.fromkeys(datasets))
    union = frames[next(iter(frames))].index
    for frame in frames.values():
        union = union.union(frame.index)
    candidates = [name for name in requested if name in union]

    missing = {}
    for estimator in estimators:
        gaps = set()
        for frame in frames.values():
            column = frame.reindex(candidates)[estimator]
            gaps.update(column.index[column.isna()])
        missing[estimator] = [name for name in candidates if name in gaps]
    return missing


def _excluded_html(missing, reasons, common, dropped) -> str:
    """Render a one-line summary of what each estimator is missing."""
    lookup = {
        (row.estimator, row.dataset): row.reason for row in reasons.itertuples()
    }

    lines = []
    for estimator, gaps in missing.items():
        if not gaps:
            continue
        # group the datasets by reason, preserving the order they appear in
        grouped: dict[str, list[str]] = {}
        for dataset in gaps:
            reason = lookup.get((estimator, dataset), "reason not recorded")
            grouped.setdefault(reason, []).append(dataset)
        summary = "; ".join(
            f"{escape(', '.join(names))} ({escape(reason)})"
            for reason, names in grouped.items()
        )
        lines.append(f"<li><b>{escape(estimator)}</b> &mdash; {summary}</li>")

    if not lines and not dropped["unknown"]:
        return ""

    parts = [
        "<h2>Missing results</h2>",
        f'<ul class="missing">{"".join(lines)}</ul>',
        f'<p class="note">Scoring uses the {len(common)} datasets every '
        "estimator completed, so a dataset any one of them is missing is left "
        "out for all. Reasons are from the job logs of these runs.</p>",
    ]
    if dropped["unknown"]:
        parts.append(
            f'<p class="note">{len(dropped["unknown"])} requested dataset(s) '
            f"have no results from any estimator: <code>"
            f"{escape(', '.join(dropped['unknown']))}</code>.</p>"
        )
    return "".join(parts)


def _describe_datasets(datasets) -> str | None:
    """Name a well-known dataset list, so the snippet can reproduce the page.

    Returns the expression that recreates ``datasets``, or None when it is not a
    list we can name.
    """
    from aeon.datasets import tsc_datasets

    requested = sorted(dict.fromkeys(datasets))
    for name in ["multiverse_core", "multiverse2026", "eeg2026", "UEA"]:
        known = getattr(tsc_datasets, name, None)
        if known is not None and requested == sorted(known):
            return f"sorted({name})"
    return None


def _snippet_html(
    datasets_expr,
    estimators,
    metrics,
    sort_by,
    max_cd_estimators,
    n_datasets,
    critical_difference,
) -> str:
    """Render the call that reproduces this page, as a copyable code block."""
    imports = ["from multiverse.experiments.tables import leaderboard"]
    if datasets_expr is None:
        datasets_expr = f"[...]  # {n_datasets} datasets"
    if "multiverse_core" in datasets_expr:
        imports.insert(0, "from aeon.datasets.tsc_datasets import multiverse_core")

    estimator_list = ", ".join(f'"{name}"' for name in estimators)
    metric_list = ", ".join(f'"{name}"' for name in metrics)
    lines = [
        *imports,
        "",
        "leaderboard(",
        f"    datasets={datasets_expr},",
        f"    estimators=[{estimator_list}],",
        f"    metrics=[{metric_list}],",
        f'    sort_by="{sort_by}",',
        *(
            [f"    critical_difference=True,",
             f"    max_cd_estimators={max_cd_estimators},"]
            if critical_difference
            else []
        ),
        ")",
    ]
    code = escape("\n".join(lines))
    return (
        "<h2>Reproducing this page</h2>"
        f"<pre><code>{code}</code></pre>"
        '<p class="note">Or <code>python -m multiverse.experiments.tables</code> '
        "to rebuild it with the defaults.</p>"
    )


def _load_all(estimators, metrics, results_dir) -> dict[str, pd.DataFrame]:
    """Load every requested metric for every estimator.

    Returns one frame per metric, indexed by dataset with one column per
    estimator. A metric is skipped if any estimator lacks a file for it, so the
    table never mixes different estimator sets across columns.
    """
    frames = {}
    for metric in metrics:
        try:
            frames[metric] = pd.DataFrame(
                {
                    estimator: load_metric(estimator, metric, results_dir)
                    for estimator in estimators
                }
            )
        except FileNotFoundError:
            continue
    if not frames:
        raise ValueError(
            f"none of {', '.join(metrics)} is available for all of "
            f"{', '.join(estimators)}"
        )
    return frames


def _common_datasets(frames, datasets) -> tuple[pd.Index, dict[str, list[str]]]:
    """Find the datasets present for every estimator on every metric.

    Every column of the summary table then describes the same set of problems,
    which is what makes the average scores and ranks comparable across metrics.
    """
    requested = list(dict.fromkeys(datasets))
    union = frames[next(iter(frames))].index
    for frame in frames.values():
        union = union.union(frame.index)

    unknown = [name for name in requested if name not in union]
    keep = pd.Index([name for name in requested if name in union])

    for frame in frames.values():
        keep = keep.intersection(frame.reindex(keep).dropna().index)

    if len(keep) == 0:
        raise ValueError(
            "no dataset has results for every estimator on every metric"
        )

    kept = set(keep)
    incomplete = [
        name for name in requested if name in union and name not in kept
    ]
    # restore the requested order, which intersection does not preserve
    return pd.Index([name for name in requested if name in kept]), {
        "unknown": unknown,
        "incomplete": incomplete,
    }


def _summary_table_html(summary, metrics, decimals) -> str:
    """Render the leaderboard: one row per estimator, score and rank per metric."""
    group_head = "".join(
        f'<th colspan="2" class="grp">{escape(METRIC_LABELS.get(m, m))}'
        f'{" &darr;" if m in LOWER_IS_BETTER else ""}</th>'
        for m in metrics
    )
    # Each header sorts its column. "first" is the direction that puts the best
    # value on top, so one click always gives the useful order: ascending for
    # ranks and for lower-is-better scores, descending otherwise.
    sub_head = "".join(
        f'<th class="sep sortable" data-col="{2 * i + 2}" '
        f'data-first="{"asc" if m in LOWER_IS_BETTER else "desc"}">Score</th>'
        f'<th class="sortable" data-col="{2 * i + 3}" data-first="asc">Rank</th>'
        for i, m in enumerate(metrics)
    )

    best_score = {
        m: (
            summary[(m, "score")].min()
            if m in LOWER_IS_BETTER
            else summary[(m, "score")].max()
        )
        for m in metrics
    }
    best_rank = {m: summary[(m, "rank")].min() for m in metrics}

    rows = []
    for position, estimator in enumerate(summary.index, 1):
        cells = []
        for m in metrics:
            score = summary.at[estimator, (m, "score")]
            rank = summary.at[estimator, (m, "rank")]
            places = 1 if abs(score) >= 1000 else decimals
            score_class = "sep best" if np.isclose(score, best_score[m]) else "sep"
            rank_class = ' class="best"' if np.isclose(rank, best_rank[m]) else ""
            cells.append(f'<td class="{score_class}">{score:,.{places}f}</td>')
            cells.append(f"<td{rank_class}>{rank:.2f}</td>")
        rows.append(
            f'<tr><td class="pos">{position}</td>'
            f"<td>{escape(str(estimator))}</td>{''.join(cells)}</tr>"
        )

    return (
        '<div class="scroll"><table id="leaderboard">'
        '<thead><tr><th rowspan="2" class="pos">#</th>'
        '<th rowspan="2" class="sortable" data-col="1" '
        f'data-first="asc" data-type="text">Estimator</th>{group_head}</tr>'
        f"<tr>{sub_head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def leaderboard(
    datasets,
    estimators,
    metrics=None,
    sort_by: str = "accuracy",
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    output_path: Path | str | None = None,
    title: str | None = None,
    critical_difference: bool = False,
    alpha: float = 0.1,
    decimals: int = 4,
    max_cd_estimators: int | None = 6,
    datasets_expr: str | None = None,
) -> Path:
    """Write a single-table HTML leaderboard.

    The table has one row per estimator and, for each metric, its average score
    and its average rank over the datasets used. Only datasets with a result for
    every estimator on every metric are included, so every column describes the
    same set of problems. Datasets that are dropped are listed on the page.

    Parameters
    ----------
    datasets : list of str
        Datasets to consider, for example
        ``aeon.datasets.tsc_datasets.multiverse_core``.
    estimators : list of str
        Estimator names, matching their results directories.
    metrics : list of str or None
        Metrics to report, in column order. Defaults to every metric in
        :data:`METRIC_LABELS`. Any metric a given estimator has no file for is
        skipped. Metrics in :data:`LOWER_IS_BETTER` are ranked ascending, the
        rest descending.
    sort_by : str, default="accuracy"
        Metric whose average rank orders the rows, best first. Also the metric
        the critical difference diagram is drawn for.
    results_dir : Path or str
        Directory holding one sub-directory per estimator.
    output_path : Path or str or None
        Where to write the page. Defaults to ``<results_dir>/leaderboard.html``.
    title : str or None
        Page heading.
    critical_difference : bool, default=False
        Whether to add a critical difference diagram below the table. Off by
        default: on these results the omnibus Friedman test does not reject over
        the leading estimators, so the diagram is a single clique and adds
        nothing the table does not already show. ``alpha`` and
        ``max_cd_estimators`` apply only when this is on.
    alpha : float, default=0.1
        Significance level for the critical difference diagram.
    decimals : int, default=4
        Decimal places for scores. Values of 1000 or more use one place.
    max_cd_estimators : int or None, default=6
        Restrict the critical difference diagram to this many best-ranked
        estimators on ``sort_by``. The table is never truncated. Every statistic
        in the diagram, the ranks, the omnibus test and the corrected alpha, is
        then computed over that subset alone.

        Note that truncating changes the statistics, it does not just hide rows.
        The diagram begins with an omnibus Friedman test over the estimators
        shown, and dropping the weakest ones compresses the spread of ranks.
        That can take the Friedman test from rejecting to not rejecting, and
        when it does not reject, aeon puts every estimator in a single clique
        and runs no pairwise tests at all. Read a truncated diagram as a
        statement about that subset only.
    datasets_expr : str or None
        How ``datasets`` was written, for the "Reproducing this page" snippet,
        for example ``"sorted(multiverse_core)"``. When None, a well-known aeon
        dataset list is recognised automatically. Only affects that snippet.

    Returns
    -------
    Path
        The file written.
    """
    estimators = list(dict.fromkeys(estimators))
    if not estimators:
        raise ValueError("at least one estimator is required")

    metrics = list(METRIC_LABELS) if metrics is None else list(dict.fromkeys(metrics))
    frames = _load_all(estimators, metrics, results_dir)
    metrics = [metric for metric in metrics if metric in frames]

    if sort_by not in frames:
        raise ValueError(
            f"sort_by={sort_by!r} is not among the available metrics: "
            f"{', '.join(metrics)}"
        )

    common, dropped = _common_datasets(frames, datasets)
    reasons = load_missing_reasons(results_dir)
    # keep the unrestricted frames: the missing-results section needs the gaps
    # that restricting to the common datasets removes
    all_frames = frames
    frames = {metric: frame.reindex(common) for metric, frame in frames.items()}

    columns = {}
    for metric in metrics:
        scores = frames[metric]
        ranks = scores.rank(axis=1, ascending=metric in LOWER_IS_BETTER)
        columns[(metric, "score")] = scores.mean()
        columns[(metric, "rank")] = ranks.mean()
    summary = pd.DataFrame(columns).sort_values((sort_by, "rank"))
    missing = _missing_by_estimator(
        all_frames, list(summary.index), common, datasets
    )

    sort_label = METRIC_LABELS.get(sort_by, sort_by)
    title = title or "Multiverse leaderboard"

    parts = [
        f"<h1>{escape(title)}</h1>",
        f'<p class="sub">{len(summary)} estimators on {len(common)} datasets '
        f"&middot; {len(metrics)} metrics &middot; ordered by average "
        f"{escape(sort_label.lower())} rank &middot; built "
        f"{date.today().isoformat()}</p>",
        _summary_table_html(summary, metrics, decimals),
        '<p class="note">Average score and average rank over the '
        f"{len(common)} datasets with results for every estimator on every "
        "metric. Best in each column is highlighted. Metrics marked &darr; are "
        "better when lower.</p>",
    ]

    if critical_difference:
        cd_order = summary.index
        cd_scores = frames[sort_by][cd_order]
        if max_cd_estimators is not None:
            cd_scores = cd_scores[cd_order[:max_cd_estimators]]
        png = _critical_difference_png(cd_scores, sort_by in LOWER_IS_BETTER, alpha)

        parts.append(
            f"<h2>Critical difference diagram: {escape(sort_label.lower())}</h2>"
        )
        if png:
            scope = (
                f"the {len(cd_scores.columns)} best-ranked of {len(summary)} "
                "estimators"
                if len(cd_scores.columns) < len(summary)
                else f"all {len(summary)} estimators"
            )
            parts.append(
                '<figure><img alt="Critical difference diagram for '
                f'{escape(sort_label)}" src="data:image/png;base64,{png}"></figure>'
                f'<p class="note">Showing {scope}. Pairwise Wilcoxon signed-rank '
                f"tests with Holm correction at alpha={alpha}. Estimators joined "
                "by a bar are not significantly different. Ranks are computed "
                "among the estimators shown, so they need not match the table "
                "above.</p>"
            )
        else:
            parts.append(
                '<p class="note">No diagram: it needs at least two estimators '
                "and enough variation between them.</p>"
            )

    parts.append(_excluded_html(missing, reasons, common, dropped))
    parts.append(
        _snippet_html(
            datasets_expr if datasets_expr is not None else _describe_datasets(datasets),
            list(summary.index),
            metrics,
            sort_by,
            max_cd_estimators,
            len(common),
            critical_difference,
        )
    )

    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main>{''.join(parts)}</main>"
        f"<script>{_SCRIPT}</script></body></html>"
    )

    output_path = (
        Path(results_dir) / "leaderboard.html"
        if output_path is None
        else Path(output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


def leaderboard_markdown(
    datasets,
    estimators,
    metrics=None,
    sort_by: str = "accuracy",
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    decimals: int = 4,
) -> str:
    """Return the leaderboard as a Markdown table.

    A narrower view than :func:`leaderboard`, meant for a README, where GitHub
    renders Markdown but shows HTML as source. Each metric contributes a score
    column, and only ``sort_by`` also gets its average rank, since a table with
    a rank beside every score is too wide to read on a repository page.

    Parameters
    ----------
    datasets : list of str
        Datasets to consider.
    estimators : list of str
        Estimator names, matching their results directories.
    metrics : list of str or None
        Metrics to show as score columns. Defaults to every metric available.
    sort_by : str, default="accuracy"
        Metric whose average rank orders the rows and is shown as a column.
    results_dir : Path or str
        Directory holding one sub-directory per estimator.
    decimals : int, default=4
        Decimal places for scores.

    Returns
    -------
    str
        The table, followed by a line saying what it covers.
    """
    estimators = list(dict.fromkeys(estimators))
    metrics = list(METRIC_LABELS) if metrics is None else list(dict.fromkeys(metrics))
    frames = _load_all(estimators, metrics, results_dir)
    metrics = [metric for metric in metrics if metric in frames]
    if sort_by not in frames:
        raise ValueError(f"sort_by={sort_by!r} is not available")

    common, _ = _common_datasets(frames, datasets)
    frames = {metric: frame.reindex(common) for metric, frame in frames.items()}

    ranks = frames[sort_by].rank(axis=1, ascending=sort_by in LOWER_IS_BETTER).mean()
    order = ranks.sort_values().index
    scores = {metric: frames[metric].mean() for metric in metrics}

    sort_label = METRIC_LABELS.get(sort_by, sort_by)
    # The rank column comes first because it is what orders the rows: with it at
    # the far right the table looks mis-sorted on whichever metric happens to be
    # leftmost.
    header = ["#", "Estimator", f"{sort_label} rank"] + [
        METRIC_LABELS.get(m, m) + (" &darr;" if m in LOWER_IS_BETTER else "")
        for m in metrics
    ]
    rows = ["| " + " | ".join(header) + " |",
            "|" + "---|" * len(header)]

    best = {
        m: (scores[m].min() if m in LOWER_IS_BETTER else scores[m].max())
        for m in metrics
    }
    for position, estimator in enumerate(order, 1):
        rank = ranks[estimator]
        cells = [
            str(position),
            estimator,
            f"**{rank:.2f}**" if np.isclose(rank, ranks.min()) else f"{rank:.2f}",
        ]
        for m in metrics:
            value = scores[m][estimator]
            text = f"{value:,.{1 if abs(value) >= 1000 else decimals}f}"
            cells.append(f"**{text}**" if np.isclose(value, best[m]) else text)
        rows.append("| " + " | ".join(cells) + " |")

    rows.append("")
    rows.append(
        f"Average over the {len(common)} Multiverse-core datasets with results for every "
        f"estimator on every metric, ordered by average {sort_label.lower()} rank. Best "
        "in each column in bold."
    )
    return "\n".join(rows)


def write_markdown_table(
    path: Path | str, table: str, marker: str = "LEADERBOARD"
) -> bool:
    """Replace the marked block in a Markdown file with ``table``.

    The block is delimited by ``<!-- MARKER:START -->`` and
    ``<!-- MARKER:END -->`` comments, which survive in the rendered page, so the
    table can be regenerated in place without disturbing the rest of the file.

    Returns
    -------
    bool
        True if the file was updated, False if the markers were not found.
    """
    path = Path(path)
    start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return False

    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(
        f"{head}{start}\n{table}\n{end}{tail}", encoding="utf-8"
    )
    return True


def dataset_summary(
    datasets,
    estimators,
    metric: str = "accuracy",
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    baseline: str = "Dummy",
) -> pd.DataFrame:
    """Summarise one metric per dataset rather than per estimator.

    The leaderboard answers "which estimator is best"; this answers "what is
    this dataset worth", which is the question an archive has to keep asking of
    its own problems.

    Unlike :func:`leaderboard` this does not restrict to the datasets every
    estimator has results for. A dataset only some estimators finished is still
    informative, so every dataset is kept and ``estimators`` records how many
    contributed.

    Parameters
    ----------
    datasets : list of str
        Datasets to include, in any order.
    estimators : list of str
        Estimator names, matching their results directories.
    metric : str
        Metric to summarise.
    results_dir : Path or str
        Directory holding one sub-directory per estimator.
    baseline : str
        Estimator treated as the no-skill floor, excluded from best, worst,
        median and spread. Pass None to keep it in.

    Returns
    -------
    pd.DataFrame
        One row per dataset, indexed by dataset name, with the baseline score,
        the median, best and worst over the remaining estimators, the estimator
        achieving the best, the gain over the baseline, the spread, and the
        number of estimators contributing.
    """
    frames = _load_all(list(estimators), [metric], results_dir)
    if metric not in frames:
        raise ValueError(f"no results for metric {metric!r}")
    scores = frames[metric].reindex(sorted(dict.fromkeys(datasets)))

    others = scores.drop(columns=[baseline], errors="ignore")
    lower_better = metric in LOWER_IS_BETTER
    best = others.min(axis=1) if lower_better else others.max(axis=1)
    worst = others.max(axis=1) if lower_better else others.min(axis=1)

    # idxmin/idxmax raise on an all-NaN row, so only ask where there is a value.
    winner = pd.Series(pd.NA, index=others.index, dtype=object)
    present = others.notna().any(axis=1)
    if present.any():
        rows = others[present]
        winner[present] = rows.idxmin(axis=1) if lower_better else rows.idxmax(axis=1)

    summary = pd.DataFrame(
        {
            "baseline": scores[baseline] if baseline in scores else np.nan,
            "median": others.median(axis=1),
            "best": best,
            "worst": worst,
            "best_estimator": winner,
            "estimators": others.notna().sum(axis=1),
        }
    )
    # The results files carry a "Resamples:" header, which pandas takes as the
    # index name and would otherwise appear as the first column heading.
    summary.index.name = "dataset"
    # Gain is how much skill the best estimator found beyond the baseline;
    # spread is how much the choice of estimator mattered. Reporting a single
    # range would conflate the two, and since the baseline is almost always the
    # weakest entry that range would just restate the gain.
    summary["gain"] = (
        summary["baseline"] - summary["best"]
        if lower_better
        else summary["best"] - summary["baseline"]
    )
    summary["spread"] = (
        summary["worst"] - summary["best"]
        if lower_better
        else summary["best"] - summary["worst"]
    )
    return summary


def _dataset_table_html(summary, decimals) -> str:
    """Render the per-dataset table, one row per dataset."""
    head = (
        '<th class="sortable" data-col="0" data-first="asc" data-type="text">'
        "Dataset</th>"
        '<th class="sortable" data-col="1" data-first="asc">Dummy</th>'
        '<th class="sortable" data-col="2" data-first="desc">Median</th>'
        '<th class="sortable" data-col="3" data-first="desc">Best</th>'
        '<th class="sortable" data-col="4" data-first="asc" data-type="text">'
        "Best estimator</th>"
        '<th class="sortable" data-col="5" data-first="asc">Gain over dummy</th>'
        '<th class="sortable" data-col="6" data-first="desc">Spread</th>'
        '<th class="sortable" data-col="7" data-first="desc">Estimators</th>'
    )

    def number(value):
        return "&mdash;" if pd.isna(value) else f"{value:.{decimals}f}"

    rows = []
    for dataset, row in summary.iterrows():
        # Two flags worth seeing at a glance: nothing beat the baseline by much,
        # and everything solves it. Both make a dataset weak at separating
        # estimators, for opposite reasons.
        classes = []
        if pd.notna(row["gain"]) and row["gain"] <= NO_SIGNAL_GAIN:
            classes.append("nosignal")
        if pd.notna(row["best"]) and row["best"] >= SATURATED_BEST:
            classes.append("saturated")
        attribute = f' class="{" ".join(classes)}"' if classes else ""
        winner = (
            "&mdash;"
            if pd.isna(row["best_estimator"])
            else escape(str(row["best_estimator"]))
        )
        rows.append(
            f"<tr{attribute}><td>{escape(str(dataset))}</td>"
            f'<td>{number(row["baseline"])}</td>'
            f'<td>{number(row["median"])}</td>'
            f'<td class="best">{number(row["best"])}</td>'
            f"<td>{winner}</td>"
            f'<td>{number(row["gain"])}</td>'
            f'<td>{number(row["spread"])}</td>'
            f'<td>{int(row["estimators"])}</td></tr>'
        )

    return (
        '<div class="scroll"><table id="leaderboard">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def dataset_page(
    datasets,
    estimators,
    metric: str = "accuracy",
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    baseline: str = "Dummy",
    sort_by: str = "gain",
    output_path: Path | str | None = None,
    title: str | None = None,
    decimals: int = 4,
) -> Path:
    """Write a self-contained page summarising one metric per dataset.

    Parameters
    ----------
    datasets : list of str
        Datasets to include.
    estimators : list of str
        Estimator names, matching their results directories.
    metric : str
        Metric to summarise.
    results_dir : Path or str
        Directory holding one sub-directory per estimator.
    baseline : str
        Estimator treated as the no-skill floor.
    sort_by : str
        Column to order rows by, one of the columns of
        :func:`dataset_summary`. The default puts the datasets where the best
        estimator gained least over the baseline at the top, because those are
        the ones worth looking at.
    output_path : Path or str, optional
        Where to write. Defaults to ``datasets.html`` beside the results.
    title : str, optional
        Page heading.
    decimals : int
        Decimal places for scores.

    Returns
    -------
    Path
        The file written.
    """
    summary = dataset_summary(datasets, estimators, metric, results_dir, baseline)
    if sort_by not in summary.columns:
        raise ValueError(f"sort_by={sort_by!r} is not a column")
    ascending = sort_by not in {"median", "best", "worst", "spread", "estimators"}
    summary = summary.sort_values(sort_by, ascending=ascending, na_position="last")

    label = METRIC_LABELS.get(metric, metric)
    title = title or f"Multiverse datasets: {label.lower()}"
    scored = int((summary["estimators"] > 0).sum())
    no_signal = int(summary["gain"].le(NO_SIGNAL_GAIN).sum())
    saturated = int(summary["best"].ge(SATURATED_BEST).sum())

    parts = [
        f"<h1>{escape(title)}</h1>",
        f'<p class="sub">{scored} datasets &middot; {escape(label.lower())}'
        f" &middot; best of up to {int(summary['estimators'].max())} estimators"
        f" against the {escape(baseline)} baseline &middot; built "
        f"{date.today().isoformat()}</p>",
        _dataset_table_html(summary, decimals),
        '<p class="note">One row per dataset. <strong>Dummy</strong> is the '
        "no-skill floor. <strong>Median</strong>, <strong>best</strong> and "
        "<strong>spread</strong> are over the other estimators, so the baseline "
        "cannot flatter them. <strong>Gain over dummy</strong> is best minus "
        "dummy, how much skill was found at all; <strong>spread</strong> is "
        "best minus worst, how much the choice of estimator mattered. The two "
        "answer different questions, and a single range would conflate them.</p>",
        f'<p class="note">{no_signal} of {scored} datasets gained '
        f"{NO_SIGNAL_GAIN:.2f} or less over the baseline (shaded amber) and "
        f"{saturated} have a best of {SATURATED_BEST:.2f} or more (shaded "
        "green). Both separate estimators poorly, for opposite reasons. Best is "
        "a maximum over many estimators, so it is optimistic by construction: "
        "read it as what the archive can currently do on a problem, not as what "
        "any one method delivers.</p>",
    ]

    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_STYLE}"
        "tr.nosignal td { background: rgba(214, 158, 46, .16); }"
        "tr.saturated td { background: rgba(56, 161, 105, .14); }"
        "</style></head>"
        f"<body><main>{''.join(parts)}</main>"
        f"<script>{_SCRIPT}</script></body></html>"
    )

    output_path = (
        Path(results_dir) / "datasets.html"
        if output_path is None
        else Path(output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


def dataset_markdown(
    datasets,
    estimators,
    metric: str = "accuracy",
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    baseline: str = "Dummy",
    sort_by: str = "gain",
    decimals: int = 4,
) -> str:
    """Return the per-dataset summary as a Markdown table."""
    summary = dataset_summary(datasets, estimators, metric, results_dir, baseline)
    ascending = sort_by not in {"median", "best", "worst", "spread", "estimators"}
    summary = summary.sort_values(sort_by, ascending=ascending, na_position="last")

    header = [
        "Dataset", "Dummy", "Median", "Best", "Best estimator",
        "Gain over dummy", "Spread", "Estimators",
    ]
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    def number(value):
        return "&mdash;" if pd.isna(value) else f"{value:.{decimals}f}"

    for dataset, row in summary.iterrows():
        winner = (
            "&mdash;"
            if pd.isna(row["best_estimator"])
            else str(row["best_estimator"])
        )
        cells = [
            str(dataset),
            number(row["baseline"]),
            number(row["median"]),
            f'**{number(row["best"])}**',
            winner,
            number(row["gain"]),
            number(row["spread"]),
            str(int(row["estimators"])),
        ]
        rows.append("| " + " | ".join(cells) + " |")

    rows.append("")
    rows.append(
        f"{METRIC_LABELS.get(metric, metric)} per dataset. Median, best and "
        f"spread are over the estimators other than {baseline}. Gain over dummy "
        "is best minus dummy; spread is best minus worst."
    )
    return "\n".join(rows)


def main() -> None:
    """Build the Multiverse-core leaderboard.

    Uses every estimator with results in the repository, including the Dummy
    baseline, over the Multiverse-core datasets all of them have results for.

    DisjointCNN-Aeon is held back. Those results are around 20 accuracy points
    below the authors' published numbers on all 23 shared datasets, because
    aeon's network applies a Permute after the final block and its pooling then
    reduces the wrong axes, leaving the classifier head one feature instead of
    64 (aeon issue #3775). They are kept as evidence for that issue rather than
    deleted, but listing them would read as a claim about the method. The
    Multiverse port of the same method reports under DisjointCNN.
    """
    from aeon.datasets.tsc_datasets import multiverse_core

    datasets = sorted(multiverse_core)
    estimators = available_estimators(exclude=("DisjointCNN-Aeon",))
    print(f"estimators: {', '.join(estimators)}")

    path = leaderboard(
        datasets,
        estimators,
        sort_by="accuracy",
        title="Multiverse-core leaderboard",
    )
    print(f"wrote {path}")

    datasets_path = dataset_page(
        datasets,
        estimators,
        metric="accuracy",
        title="Multiverse-core datasets: accuracy",
    )
    print(f"wrote {datasets_path}")

    table = leaderboard_markdown(datasets, estimators, sort_by="accuracy")
    readme = Path(__file__).resolve().parents[2] / "README.md"
    if write_markdown_table(readme, table):
        print(f"updated the table in {readme}")
    else:
        print(f"no LEADERBOARD markers in {readme}; Markdown table not written")


if __name__ == "__main__":
    main()
