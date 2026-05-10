# Dyana Output Table Format

This document describes the current plain-text table format used by Dyana's
migrated analyses.

## Scope

This format currently applies to the supported shared-output path used by:

- RDF
- density
- ADF
- neighbor count

## File Structure

The standard layout is:

1. optional metadata/comment lines
2. one commented column-header line
3. numeric data rows

Example:

```text
# analysis: rdf
# r/Angstrom g(r) N(r)
0.005000             0.000000             0.000000
0.015000             0.000000             0.000000
```

## Formatting Rules

- comment lines start with `# `
- the column-header line is written as `# <header1> <header2> ...`
- data rows use spaces only, never tabs
- floats use `%.6f`
- integers are written as plain integers
- the first data column has no leading spaces
- the first data column is left-aligned within its field
- later data columns are right-aligned
- the default data-column width is `16`
- headers and data use ASCII-friendly text such as `Angstrom`

## Writer API

The primary writer API is:

- `write_table(...)`

The histogram helper:

- `write_histogram_1d(...)`

is a thin adapter that converts a one-dimensional histogram into rows and then
delegates to `write_table(...)`.
