"""Sudoku grid primitives shared across the package.

A leaf module (imports only the stdlib) so :mod:`loopscape.base` and the task
codecs can use ``Grid``/``parse_puzzle``/``is_valid_solution`` without pulling
in any model code.  :mod:`loopscape.fprm.inference` re-exports these names for
backwards compatibility.
"""
from __future__ import annotations

from typing import List, Union

Grid = List[List[int]]


def parse_puzzle(puzzle: Union[str, Grid]) -> Grid:
    """Parse a puzzle into a 9x9 grid of ints (0 = blank).

    Accepts an 81-char string ('.', '0' or space = blank, '1'-'9' = digit) or a
    9x9 / flat list of ints.
    """
    if isinstance(puzzle, str):
        s = "".join(puzzle.split())  # drop whitespace/newlines
        if len(s) != 81:
            raise ValueError(f"String puzzle must have 81 cells, got {len(s)}")
        vals = [0 if c in "._0" else int(c) for c in s]
    else:
        flat = [c for row in puzzle for c in row] if puzzle and isinstance(puzzle[0], (list, tuple)) else list(puzzle)
        if len(flat) != 81:
            raise ValueError(f"Puzzle must have 81 cells, got {len(flat)}")
        vals = [int(c) for c in flat]
    return [vals[r * 9:(r + 1) * 9] for r in range(9)]


def format_grid(grid: Grid) -> str:
    """Pretty 9x9 grid with box separators; 0/blank shown as '.'."""
    lines = []
    for r in range(9):
        if r % 3 == 0 and r > 0:
            lines.append("------+-------+------")
        cells = []
        for c in range(9):
            if c % 3 == 0 and c > 0:
                cells.append("|")
            v = grid[r][c]
            cells.append(str(v) if 1 <= v <= 9 else ".")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def is_valid_solution(grid: Grid) -> bool:
    """True iff every row, column and 3x3 box is a permutation of 1..9."""
    want = set(range(1, 10))
    for r in range(9):
        if set(grid[r]) != want:
            return False
    for c in range(9):
        if {grid[r][c] for r in range(9)} != want:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = {grid[br + i][bc + j] for i in range(3) for j in range(3)}
            if box != want:
                return False
    return True
