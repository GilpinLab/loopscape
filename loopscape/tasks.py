"""Task abstraction: puzzle encoding / decoding / validation per reasoning task.

The unified solver API (:mod:`loopscape.base`) runs a model's recurrence given an
initial latent state and reads out a decoded grid.  Everything *task-specific* --
how a puzzle string maps to input tokens, how logits map back to a grid, what
"solved" means, and the board geometry -- lives here, so the solvers themselves
stay task-agnostic and the batched initial-state API is identical across tasks.

Two tasks are provided:

* ``SUDOKU`` : 9x9 grid, vocab 11.  Cells are ``digit + 1`` (blank 0 -> token 1,
  digits 1..9 -> tokens 2..10); decode is ``argmax - 1``.  A solve is correct iff
  every given is preserved and the grid is a valid Sudoku.
* ``MAZE``   : 30x30 grid, vocab 6.  Cells are ``charset.index(ch) + 1`` over the
  charset ``"# SGo"`` (wall ``#`` -> 1, open `` `` -> 2, start ``S`` -> 3, goal
  ``G`` -> 4, path ``o`` -> 5; pad id 0 is unused).  This matches the HRM/EqR maze
  dataset tokenisation exactly.  A solve is correct iff it reproduces the
  reference solution (the maze has a unique shortest path, so exact match to the
  labelled path is the criterion).

Look up a task by name with :func:`get_task`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

# A decoded grid is a list of rows of ints (task-specific cell alphabet).
Grid = List[List[int]]
PuzzleLike = Union[str, Grid, Sequence[int]]


def _flatten(puzzle: PuzzleLike, n_cells: int) -> List[int]:
    """Normalise a grid / flat list to a flat list of ``n_cells`` ints."""
    if puzzle and isinstance(puzzle[0], (list, tuple)):
        flat = [c for row in puzzle for c in row]
    else:
        flat = list(puzzle)
    if len(flat) != n_cells:
        raise ValueError(f"Puzzle must have {n_cells} cells, got {len(flat)}")
    return [int(c) for c in flat]


@dataclass(frozen=True)
class Task:
    """Task-specific puzzle codec + validator + geometry.

    Attributes
    ----------
    name        : task key ("sudoku" | "maze").
    board       : (rows, cols) of the grid.
    vocab_size  : model vocabulary size (includes pad id 0).
    char_to_cell: parse one character of a string puzzle to its integer cell value.
    cell_to_char: render one integer cell value to a display character.
    cell_to_token / token_to_cell : the encode/decode shift between cell values and
        model token ids (Sudoku: token = cell + 1; Maze: token = cell (already ids)).
    is_solved   : correctness check ``(puzzle, solution) -> bool``.
    """

    name: str
    board: Tuple[int, int]
    vocab_size: int
    _char_to_cell: Callable[[str], int]
    _cell_to_char: Callable[[int], str]
    _cell_to_token: Callable[[int], int]
    _token_to_cell: Callable[[int], int]
    _is_solved: Callable[["Task", PuzzleLike, Grid], bool]
    # Sudoku strings may carry cosmetic spaces/newlines; maze strings use the space
    # character as a *real* cell (open corridor), so only newlines are stripped there.
    strip_all_whitespace: bool = True

    # ---- geometry -----------------------------------------------------------
    @property
    def rows(self) -> int:
        return self.board[0]

    @property
    def cols(self) -> int:
        return self.board[1]

    @property
    def seq_len(self) -> int:
        return self.board[0] * self.board[1]

    # ---- parsing / (de)serialisation ---------------------------------------
    def parse(self, puzzle: PuzzleLike) -> Grid:
        """Parse a string / grid / flat list into a ``rows x cols`` grid of cells."""
        if isinstance(puzzle, str):
            if self.strip_all_whitespace:
                s = "".join(puzzle.split())          # drop cosmetic spaces + newlines
            else:
                s = puzzle.replace("\n", "").replace("\r", "")  # keep spaces (real cells)
            if len(s) != self.seq_len:
                raise ValueError(f"String puzzle must have {self.seq_len} cells, got {len(s)}")
            vals = [self._char_to_cell(c) for c in s]
        else:
            vals = _flatten(puzzle, self.seq_len)
        return [vals[r * self.cols:(r + 1) * self.cols] for r in range(self.rows)]

    def encode(self, puzzle: PuzzleLike) -> torch.Tensor:
        """Puzzle -> int32 input tokens of shape ``[1, seq_len]``."""
        grid = self.parse(puzzle)
        flat = [grid[r][c] for r in range(self.rows) for c in range(self.cols)]
        tokens = [self._cell_to_token(v) for v in flat]
        return torch.tensor(tokens, dtype=torch.int32).unsqueeze(0)

    def decode(self, logits: torch.Tensor) -> Grid:
        """``[1, seq_len, vocab]`` logits -> one decoded grid."""
        pred = logits.argmax(dim=-1).squeeze(0).tolist()
        cells = [self._token_to_cell(t) for t in pred]
        return [cells[r * self.cols:(r + 1) * self.cols] for r in range(self.rows)]

    def decode_batch(self, logits: torch.Tensor) -> List[Grid]:
        """``[B, seq_len, vocab]`` logits -> list of B decoded grids.

        Vectorised: token->cell is applied through a vocab-sized lookup table
        instead of a per-cell Python call, which matters when decoding every
        step of large batched sweeps.
        """
        preds = logits.argmax(dim=-1).cpu()             # [B, seq_len]
        lut = torch.tensor([self._token_to_cell(t) for t in range(self.vocab_size)])
        cells = lut[preds.long()].view(len(preds), self.rows, self.cols)
        return cells.tolist()

    def format_grid(self, grid: Grid) -> str:
        """Render a grid to a display string (rows joined by newlines)."""
        return "\n".join("".join(self._cell_to_char(v) for v in row) for row in grid)

    def is_solved(self, puzzle: PuzzleLike, solution: Grid) -> bool:
        return self._is_solved(self, puzzle, solution)


# ------------------------------------------------------------------ Sudoku task

def _sudoku_char_to_cell(c: str) -> int:
    return 0 if c in "._0" else int(c)


def _sudoku_cell_to_char(v: int) -> str:
    return str(v) if 1 <= v <= 9 else "."


def _sudoku_is_valid(grid: Grid) -> bool:
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
            if {grid[br + i][bc + j] for i in range(3) for j in range(3)} != want:
                return False
    return True


def _sudoku_is_solved(task: Task, puzzle: PuzzleLike, solution: Grid) -> bool:
    g = task.parse(puzzle)
    givens_ok = all(g[r][c] == 0 or g[r][c] == solution[r][c]
                    for r in range(9) for c in range(9))
    return givens_ok and _sudoku_is_valid(solution)


SUDOKU = Task(
    name="sudoku",
    board=(9, 9),
    vocab_size=11,
    _char_to_cell=_sudoku_char_to_cell,
    _cell_to_char=_sudoku_cell_to_char,
    _cell_to_token=lambda v: v + 1,      # blank 0 -> 1, digit d -> d+1
    _token_to_cell=lambda t: t - 1,
    _is_solved=_sudoku_is_solved,
)


# -------------------------------------------------------------------- Maze task
# HRM/EqR maze tokenisation: cell id = charset index + 1 (pad id 0 unused).
_MAZE_CHARSET = "# SGo"                                   # ids 1..5
_MAZE_CHAR_TO_CELL = {c: i + 1 for i, c in enumerate(_MAZE_CHARSET)}
_MAZE_CELL_TO_CHAR = {i + 1: c for i, c in enumerate(_MAZE_CHARSET)}
_MAZE_PATH_ID = _MAZE_CHAR_TO_CELL["o"]                   # 5


def _maze_char_to_cell(c: str) -> int:
    try:
        return _MAZE_CHAR_TO_CELL[c]
    except KeyError:
        raise ValueError(f"Unknown maze character {c!r}; expected one of {_MAZE_CHARSET!r}")


def _maze_cell_to_char(v: int) -> str:
    return _MAZE_CELL_TO_CHAR.get(int(v), "?")


def _maze_is_solved(task: Task, puzzle: PuzzleLike, solution: Grid) -> bool:
    """Correct iff the solution reproduces the reference (unique shortest path).

    ``puzzle`` here must be the *reference solution* (the labelled maze with its
    path marked ``o``): the maze has a unique shortest path, so exact grid match
    is the correctness criterion.  Passing the unsolved maze (no path) will simply
    report False unless the model also predicts an empty path.
    """
    ref = task.parse(puzzle)
    return ref == solution


MAZE = Task(
    name="maze",
    board=(30, 30),
    vocab_size=len(_MAZE_CHARSET) + 1,                   # 6
    _char_to_cell=_maze_char_to_cell,
    _cell_to_char=_maze_cell_to_char,
    _cell_to_token=lambda v: v,          # cells are already token ids (1..5)
    _token_to_cell=lambda t: t,
    _is_solved=_maze_is_solved,
    strip_all_whitespace=False,          # ' ' is a real (open) cell in mazes
)


def maze_shortest_path_length(maze: PuzzleLike) -> int:
    """Difficulty rating of a maze = number of cells on the shortest S->G path.

    Computed directly from the maze structure with a breadth-first search over the
    open (non-wall) cells: walls are ``#`` (cell id 1); every other cell (open,
    ``S``, ``G``, and -- if present -- an ``o`` path marker) is passable.  The
    returned length counts the nodes on the shortest path *including* both the start
    ``S`` and goal ``G`` endpoints.

    This reproduces the ``rating`` column of the Maze-Hard-1k dataset exactly (that
    rating is precisely this quantity), so it is the natural rating for the
    Maze-Unique-1k split, which ships without a rating column.  Accepts the unsolved
    maze (``question``) or the solved one (``answer``) -- both yield the same value.

    Returns ``-1`` if the goal is unreachable, or raises if the maze has no ``S``/``G``.

    This function is intentionally standalone (not folded into the ``Task`` methods
    or the dataloader); callers invoke it explicitly to assign ratings.
    """
    grid = MAZE.parse(maze)                              # 30x30 cells (ids 1..5)
    n_rows, n_cols = MAZE.board
    wall = _MAZE_CHAR_TO_CELL["#"]                       # 1
    start_id, goal_id = _MAZE_CHAR_TO_CELL["S"], _MAZE_CHAR_TO_CELL["G"]

    start = goal = None
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c] == start_id:
                start = (r, c)
            elif grid[r][c] == goal_id:
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("maze must contain both a start 'S' and a goal 'G'")

    from collections import deque
    dist = {start: 1}                                    # count nodes, so S is 1
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return dist[(r, c)]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols \
                    and grid[nr][nc] != wall and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return -1                                            # goal unreachable


# ------------------------------------------------------------------- registry

_TASKS: Dict[str, Task] = {SUDOKU.name: SUDOKU, MAZE.name: MAZE}


def get_task(task: Union[str, Task]) -> Task:
    """Resolve a task by name (case-insensitive) or pass a ``Task`` through."""
    if isinstance(task, Task):
        return task
    key = str(task).lower()
    if key not in _TASKS:
        raise ValueError(f"Unknown task {task!r}. Available: {sorted(_TASKS)}")
    return _TASKS[key]


def available_tasks() -> List[str]:
    return sorted(_TASKS)
