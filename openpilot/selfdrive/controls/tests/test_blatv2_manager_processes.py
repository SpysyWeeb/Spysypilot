from __future__ import annotations

import ast
from pathlib import Path
PROCESS_CONFIG = (
  Path(__file__).resolve().parents[3]
  / "system"
  / "manager"
  / "process_config.py"
)


def _manager_blatv2_processes(
  tree: ast.AST,
) -> dict[str, str]:
  processes: dict[str, str] = {}
  for node in ast.walk(tree):
    if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name)
      and node.func.id == "PythonProcess"
      and len(node.args) >= 3
      and isinstance(node.args[0], ast.Constant)
      and isinstance(node.args[0].value, str)
      and node.args[0].value.startswith("blatv2_")
      and isinstance(node.args[2], ast.Name)
    ):
      processes[node.args[0].value] = node.args[2].id
  return processes


def test_manager_runs_no_blatv2_background_process() -> None:
  tree = ast.parse(
    PROCESS_CONFIG.read_text(encoding="utf-8"),
    filename=str(PROCESS_CONFIG),
  )
  assert _manager_blatv2_processes(tree) == {}
