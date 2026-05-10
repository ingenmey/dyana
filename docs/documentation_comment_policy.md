# Dyana Documentation And Comment Policy

This document defines how Dyana should document code, behavior, and design.
The goal is not to maximize the amount of documentation. The goal is to keep
the project understandable, scientifically trustworthy, and easy to maintain.

The guiding rule is:

> Prefer clear code, concise comments, and stable reference documentation over
> large amounts of stale or repetitive text.

## 1. Core Principles

- Write documentation to make the supported behavior and design easy to
  understand.
- Prefer clarity and correctness over preserving old wording after refactors.
- Keep comments and docstrings short unless extra detail truly helps.
- Do not explain obvious code line by line.
- Do document scientific assumptions, invariants, and non-obvious behavior.
- Treat stale comments and stale docs as bugs.

## 2. Documentation Layers

Dyana should use several documentation layers with different purposes.

### README

The top-level [README.md](D:/python/dyana/README.md) should stay short and cover:

- what Dyana is
- how to run it
- basic development/test commands
- a short note on the current supported workflow

The README is not the place for deep design detail or full analysis reference
material.

### Planning Docs

The checklist documents in [docs](D:/python/dyana/docs) exist to track design
direction and project state:

- [analysis_framework_checklist.md](D:/python/dyana/docs/analysis_framework_checklist.md)
- [productionization_checklist.md](D:/python/dyana/docs/productionization_checklist.md)

These documents answer:

- what direction we are taking
- what is done vs still open
- what tradeoffs were chosen

They are not a substitute for stable reference documentation.

### Reference Docs

Stable user/developer reference docs should explain the current supported
behavior of the codebase, for example:

- supported trajectory formats
- the runtime topology model
- the analysis lifecycle
- output file conventions
- atom-label and selection semantics
- static vs dynamic topology behavior

Reference docs should describe how the current supported path works, not just
what we hope to change later.

### Docstrings

Docstrings are for public or shared code surfaces that other modules or future
maintainers need to understand quickly.

They should explain:

- what the class/function is for
- important arguments and return values when not obvious
- important invariants or assumptions
- any behavior that is easy to misuse

They should not repeat the function name in sentence form or narrate obvious
assignments and loops.

### Inline Comments

Inline comments should be rare and intentional.

They are appropriate for:

- non-obvious control flow
- invariants the code relies on
- scientific rationale
- temporary caveats that are still true and worth knowing
- explaining why a simpler-looking approach would be wrong

They are not appropriate for:

- restating obvious code
- describing simple assignments
- narrating every branch

### Tests

Tests are part of the behavioral specification, but they are not the primary
documentation surface.

Use tests to:

- lock in intended behavior
- cover edge cases
- encode scientific expectations numerically
- protect invariants during refactors

Do not rely on tests alone to communicate the public model or scientific
assumptions.

## 3. Comment Strategy

Dyana should prefer lightly commented code with strong names and clear
structure.

The default rule is:

- first make the code readable
- then add comments only where readability alone is not enough

Good comment targets in Dyana include:

- why canonical local ordering exists
- why a frame is skipped
- why a topology rebuild is or is not performed
- why a normalization is defined a certain way
- why a selection is resolved once and reused

Bad comment targets include:

- "increment frame index"
- "assign config value"
- "loop over molecules"

## 4. Docstring Strategy

Public/shared code should generally have docstrings. In practice, this means:

- shared runtime model classes
- shared framework classes
- shared configuration objects
- shared output-writer entry points
- trajectory reader/base interfaces

Private helpers do not need docstrings unless they are tricky, easy to misuse,
or implement an important invariant.

For Dyana, docstrings should usually be short prose, not heavy template-driven
API blocks, unless a function truly has a non-obvious signature or return
contract.

## 5. Scientific And Behavioral Contracts

The following topics must be documented explicitly somewhere stable when they
matter to supported behavior:

- atom-label semantics
- canonical local ordering
- symmetry-equivalent atom policy
- static vs dynamic topology behavior
- frame indexing conventions
- output normalization semantics
- units and output column meanings
- same-molecule exclusion behavior

These should not live only in chat history or be recoverable only by reading
tests.

## 6. Maintenance Rules

When code changes, documentation and comments should be updated in the same
change when the public behavior, invariants, or supported workflow changed.

In practice:

- if a comment becomes misleading, remove or rewrite it immediately
- if a docstring no longer matches behavior, update it immediately
- if a plan checklist no longer matches the code, realign it
- if a behavior becomes part of the supported path, move it out of "planning
  only" territory and document it in a stable reference location

Do not keep legacy wording just because it was once true.

## 7. Near-Term Documentation Priorities

Given the current state of Dyana, the highest-value documentation work is:

1. document the runtime topology model as it now exists
2. document the analysis lifecycle and config flow
3. document the supported output format conventions
4. document the labeling/selection/topology-update behavior that supported
   analyses rely on

This should happen before broad expansion into more analyses.
