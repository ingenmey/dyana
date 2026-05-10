from __future__ import annotations

import hashlib
import numbers
import re
from collections.abc import Iterable
from pathlib import Path


_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9+.-]+")
_DEFAULT_DATA_COLUMN_WIDTH = 16
_OUTPUT_DIR = Path(".")
_FORCE_OVERWRITE = False


def build_output_filename(analysis_name: str, explicit_parts: list[str] | None = None, max_length: int = 60) -> str:
    """Build an output filename from an analysis name and optional selection parts."""
    if not explicit_parts:
        return f"{analysis_name}.dat"

    explicit_stem = "_".join([analysis_name, *explicit_parts])
    if len(explicit_stem) <= max_length:
        return f"{explicit_stem}.dat"

    short_id = hashlib.sha1(explicit_stem.encode("utf-8")).hexdigest()[:8]
    return f"{analysis_name}_{short_id}.dat"


def configure_output(output_dir: str | Path = ".", force_overwrite: bool = False) -> tuple[Path, bool]:
    """Set the shared output directory and overwrite policy, returning the previous policy."""
    global _OUTPUT_DIR, _FORCE_OVERWRITE
    previous = (_OUTPUT_DIR, _FORCE_OVERWRITE)
    _OUTPUT_DIR = Path(output_dir)
    _FORCE_OVERWRITE = bool(force_overwrite)
    return previous


def restore_output(previous: tuple[Path, bool]) -> None:
    """Restore a previously captured shared output policy."""
    global _OUTPUT_DIR, _FORCE_OVERWRITE
    _OUTPUT_DIR, _FORCE_OVERWRITE = previous


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

    output_path = _prepare_output_path(filename)

    with open(output_path, "w", encoding="utf-8") as f:
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


def _prepare_output_path(filename: str | Path) -> Path:
    path = Path(filename)
    if not path.is_absolute():
        path = _OUTPUT_DIR / path

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not _FORCE_OVERWRITE:
        rotated_path, shifted_count = _rotate_existing_output(path)
        if shifted_count > 1:
            print(
                f"Output file {path.name} already existed; moved the previous file to "
                f"{rotated_path.name} and shifted {shifted_count - 1} older backup(s)."
            )
        else:
            print(f"Output file {path.name} already existed; moved the previous file to {rotated_path.name}.")

    return path


def _rotate_existing_output(path: Path) -> tuple[Path, int]:
    max_index = 0
    while _backup_path(path, max_index + 1).exists():
        max_index += 1

    for index in range(max_index, 0, -1):
        _backup_path(path, index).replace(_backup_path(path, index + 1))

    rotated_path = _backup_path(path, 1)
    path.replace(rotated_path)
    return rotated_path, max_index + 1


def _backup_path(path: Path, index: int) -> Path:
    return path.with_name(f"#{index}#{path.name}")


def _render_table_line(values: Iterable[str], widths: list[int]) -> str:
    values = list(values)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    rendered = [values[0].ljust(widths[0])]
    rendered.extend(value.rjust(width) for value, width in zip(values[1:], widths[1:]))
    return " ".join(rendered)
