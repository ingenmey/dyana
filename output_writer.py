from __future__ import annotations

import hashlib
import numbers
import re
from collections.abc import Iterable


_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9+.-]+")
_DEFAULT_DATA_COLUMN_WIDTH = 16


def build_output_filename(analysis_name: str, explicit_parts: list[str] | None = None, max_length: int = 60) -> str:
    """Build an output filename from an analysis name and optional selection parts."""
    if not explicit_parts:
        return f"{analysis_name}.dat"

    explicit_stem = "_".join([analysis_name, *explicit_parts])
    if len(explicit_stem) <= max_length:
        return f"{explicit_stem}.dat"

    short_id = hashlib.sha1(explicit_stem.encode("utf-8")).hexdigest()[:8]
    return f"{analysis_name}_{short_id}.dat"


def format_selection(labels: Iterable[str], compound_rep: str) -> str:
    """Format one label/compound selection for filenames."""
    label_part = "+".join(_sanitize_token(label) for label in labels)
    return f"{label_part}-{_sanitize_token(compound_rep)}"


def format_selection_group(selections: Iterable[tuple[Iterable[str], str]]) -> str:
    """Format a group of selections for filenames."""
    return "+".join(format_selection(labels, compound_rep) for labels, compound_rep in selections)


def write_histogram_1d(
    filename: str,
    hist,
    headers: list[str] | None = None,
    fields: list[str] | None = None,
    comment_lines: list[str] | None = None,
) -> None:
    """Write a one-dimensional histogram through the shared table writer."""
    if len(hist.bin_edges) != 1:
        raise ValueError("write_histogram_1d only supports one-dimensional histograms.")

    bin_centers = 0.5 * (hist.bin_edges[0][1:] + hist.bin_edges[0][:-1])
    if fields is None:
        fields = list(hist.data.keys())

    columns = [bin_centers] + [hist.data[field].flatten() for field in fields]
    if headers is None:
        headers = ["bin_0", *fields]
    elif len(headers) != len(columns):
        raise ValueError(f"headers must have {len(columns)} entries, got {len(headers)}.")

    rows = list(zip(*columns))
    write_table(filename, headers=headers, data=rows, comment_lines=comment_lines)


def write_table(
    filename: str,
    headers: list[str],
    data: Iterable[Iterable[object]],
    comment_lines: list[str] | None = None,
) -> None:
    """Write a plain-text numeric table using the shared Dyana format."""
    rows = [list(row) for row in data]
    ncols = len(headers)
    for row in rows:
        if len(row) != ncols:
            raise ValueError(f"All rows must have {ncols} values, got {len(row)}.")

    formatted_rows = [[_format_default(value) for value in row] for row in rows]
    widths = [_DEFAULT_DATA_COLUMN_WIDTH for _ in headers]
    for idx, header in enumerate(headers):
        widths[idx] = max(widths[idx], len(header))
    for row in formatted_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    with open(filename, "w", encoding="utf-8") as f:
        for line in comment_lines or []:
            f.write(f"# {line}\n")

        f.write("# " + _render_table_line(headers, widths) + "\n")
        for row in formatted_rows:
            f.write(_render_table_line(row, widths) + "\n")


def _format_default(value: object) -> str:
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return f"{float(value):.6f}"
    return str(value)


def _sanitize_token(value: str) -> str:
    cleaned = _SAFE_TOKEN_RE.sub("", value.replace(" ", ""))
    if not cleaned:
        raise ValueError(f"Could not build a filename token from {value!r}.")
    return cleaned


def _render_table_line(values: Iterable[str], widths: list[int]) -> str:
    values = list(values)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    rendered = [values[0].ljust(widths[0])]
    rendered.extend(value.rjust(width) for value, width in zip(values[1:], widths[1:]))
    return " ".join(rendered)
