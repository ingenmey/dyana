# Dyana Interactive Console Output

This document describes the current supported console-output style for Dyana's
interactive path.

The goal is:

- concise but informative output
- clear phase boundaries
- readable progress and result messages
- a plain-text mirrored console log for later inspection

## 1. Console Structure

The supported interactive path uses a small shared console helper in
[io_support/console.py](D:/python/dyana/io_support/console.py).

Interactive runs are structured into a few stable sections:

1. header
2. topology setup
3. available analyses / analysis setup
4. frame loop
5. run
6. results

This keeps the session readable without turning it into a long wizard.

## 2. Header

The header is intentionally short. It should identify the run context without
repeating details that will appear later.

Current header lines include:

- Dyana version
- run start time
- trajectory path
- trajectory format when known
- output directory
- console log path
- input log path when present
- prepared-setup path when present

Paths in the header are written as absolute paths so the recorded run context
remains clear even if output files are moved or inspected later from a
different working directory.

## 3. Status Message Types

The supported console vocabulary is intentionally small:

- plain text for neutral data blocks and lists
- `info` for secondary run details
- `success` for accepted actions and written outputs
- `warn` for recoverable issues and rotated files
- `error` for direct user-facing failures
- `progress` for frame-processing heartbeats

These should stay consistent across the supported interactive path.

## 4. Colors

Colors are optional and should remain lightweight.

Current usage is:

- cyan for section headers and progress
- green for success messages
- yellow for warnings
- red for errors
- dim text for secondary informational notes

The console helper automatically falls back to plain text when color is not
supported, and ANSI color is stripped from the mirrored log file.

## 5. Logging

Interactive runs now mirror shared console output into:

- `<output-dir>/dyana.log`

This file is a plain-text record of the supported interactive console
transcript. It includes:

- shared console output emitted through the console helper
- interactive prompts
- user replies

Prompt/answer logging remains separate:

- `<output-dir>/input.log`

The input log is still the replayable prompt-answer record, while `dyana.log`
is the human-readable full console transcript.

## 6. Style Guidelines

Good console output in Dyana should:

- show phase changes clearly
- report progress sparingly
- make written outputs obvious
- avoid repeating the same facts in multiple places

It should not:

- print large decorative banners
- narrate every small step
- repeat prompt text again in later synthetic summaries
- rely on color alone to communicate meaning
