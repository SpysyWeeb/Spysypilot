from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


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


def _isolated_predicate(
  tree: ast.Module,
  name: str,
):
  function = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == name
  )
  function.returns = None
  for argument in (
    *function.args.posonlyargs,
    *function.args.args,
    *function.args.kwonlyargs,
  ):
    argument.annotation = None
  module = ast.fix_missing_locations(ast.Module(
    body=[function],
    type_ignores=[],
  ))
  namespace: dict[str, object] = {}
  exec(compile(module, str(PROCESS_CONFIG), "exec"), namespace)
  return namespace[name]


def test_manager_runs_only_backfill_offroad() -> None:
  tree = ast.parse(
    PROCESS_CONFIG.read_text(encoding="utf-8"),
    filename=str(PROCESS_CONFIG),
  )
  assert _manager_blatv2_processes(tree) == {
    "blatv2_backfilld": "blatv2_offroad",
  }

  should_run = _isolated_predicate(tree, "blatv2_offroad")
  real_car = SimpleNamespace(notCar=False)
  non_car = SimpleNamespace(notCar=True)
  unused_params = object()

  assert should_run(False, unused_params, real_car) is True
  assert should_run(True, unused_params, real_car) is False
  assert should_run(False, unused_params, non_car) is False
  assert should_run(True, unused_params, non_car) is False
