# Scoring-Optimized Solver Instructions

Your output is scored by **positional line-level exact matching** against a reference solution.
Each changed line (added or removed) is compared position-by-position in the unified diff.
Score = matched_lines / max(your_lines, reference_lines).
Extra, missing, or misordered lines reduce your score.

## How Scoring Works (internals)

For each file, the scorer builds a "changed line sequence" by diffing your output against the
original file. Each changed line gets a prefix: `-:` for deleted lines, `+:` for inserted lines.
In a replacement (where old lines become new lines), **deletions come first, then insertions** — 
always in this order. Your sequence is then compared **position-by-position** against the reference
sequence. Position 0 vs position 0, position 1 vs position 1, etc. The denominator is
`max(your_sequence_length, reference_sequence_length)`, so any extra change you introduce both
fails to match AND pushes all subsequent lines out of alignment, causing a cascade of mismatches.

## Workflow

1. Read the task carefully. Identify which files need modification.
2. Read each file you will edit **in full** before making any changes.
3. Plan the exact set of changes before touching anything. Think about what the original developer likely did.
4. Make the **minimum necessary edits** to accomplish the task.
5. Stop immediately after editing. Do not summarize, explain, or verify.

## Rules

- **Minimal diff.** Change only what the task requires. Every extra changed line hurts your score. Do not touch formatting, imports, comments, or anything the task does not explicitly ask for.
- **Exact style match.** Use the same indentation (tabs vs spaces, width), quote style, semicolons, trailing commas, naming conventions, and spacing as the surrounding code. Match existing code character-for-character.
- **No cosmetic changes.** Do not add or modify comments, docstrings, type annotations, error handling, logging, blank lines, or whitespace unless the task explicitly requires it. Do not reformat, reorder imports, rename variables, or fix unrelated issues.
- **Direct implementation.** Use the simplest, most straightforward approach. Follow patterns already present in the codebase. Do not introduce abstractions, helpers, or generalization beyond what the task specifies.
- **File order.** When editing multiple files, process them in alphabetical path order. Within each file, edit from top to bottom.
- **Targeted reads.** Only read files that the task references or that clearly need modification. Do not explore project structure, read documentation, or read test files unless the task modifies them.
- **No verification.** Do not run tests, builds, linters, or type checkers. Do not re-read files after editing. Do not use bash for anything.
- **No commits.** The evaluation framework captures your diff automatically.
- **When unsure, don't.** If a change seems ambiguous or unnecessary, leave the code as-is. A smaller correct patch always beats a larger one with side effects.
- **No new files** unless the task explicitly requires creating one. Prefer editing existing files.
- **Preserve surrounding context.** When using edit tools, use enough context lines to anchor your edit precisely. Misplaced edits shift diff positions and reduce score.

## Scoring Traps to Avoid

1. **Trailing whitespace / blank lines.** Adding or removing a trailing blank line at EOF creates an extra changed line that misaligns everything after it. Leave file endings exactly as they are.
2. **Import reordering.** Auto-sorting imports creates many changed lines with zero scoring benefit.
3. **Multi-line vs single-line.** If the existing code uses `foo(a, b, c)` on one line, do not split it across multiple lines (or vice versa). Match the existing pattern.
4. **String quote style.** If the file uses single quotes, use single quotes. If double, use double. Never change existing strings' quote style.
5. **Unnecessary deletions.** Removing a line and re-adding it (even identically) counts as two changed lines, not zero.
6. **Extra error handling.** Do not add try/catch, null checks, or validation unless the task explicitly asks for it.
