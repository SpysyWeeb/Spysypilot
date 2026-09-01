#!/usr/bin/env python3
"""Isolation check (shadow_learner_design.md tests_required item 2): a
process-level analog of a stub-and-diff actuator test, appropriate here
because the observer has no call site to stub -- it is a standalone process
with no import of, or write path to, carControl/actuators.

Static/AST-based rather than a live sim: walks the two new modules' own
import statements and attribute accesses (not the full transitive dependency
closure -- that would also flag legitimate type-only imports like
opendbc.car.structs.car) and asserts neither ever names a controls-authority
module or the 'actuators' (write) attribute -- only 'actuatorsOutput' (the
already-applied-torque telemetry the design explicitly reads as hMeasured).
"""
import ast
import os

LOCATIOND_DIR = os.path.dirname(os.path.dirname(__file__))
OBSERVER_PATH = os.path.join(LOCATIOND_DIR, "rack_effort_observer.py")
CLASSIFIER_PATH = os.path.join(LOCATIOND_DIR, "rack_effort_classifier.py")

# Modules that own a real path to commanding the wheel; RESO must never import any of these.
FORBIDDEN_IMPORT_PREFIXES = (
  "openpilot.selfdrive.controls",  # controlsd.py, latcontrol_*.py: the actual torque authority
  "openpilot.selfdrive.car.card",  # the process that sends CAN
)


def _parse(path):
  with open(path) as f:
    return ast.parse(f.read(), filename=path)


def _imported_modules(tree):
  mods = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      mods.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
      mods.add(node.module)
  return mods


def _attribute_names(tree):
  return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def test_no_controls_authority_import():
  mods = _imported_modules(_parse(OBSERVER_PATH)) | _imported_modules(_parse(CLASSIFIER_PATH))
  hits = [m for m in mods if any(m == p or m.startswith(p + ".") for p in FORBIDDEN_IMPORT_PREFIXES)]
  assert not hits, f"rack effort observer imports a torque-authority module: {hits}"


def test_never_accesses_actuators_write_path():
  attrs = _attribute_names(_parse(OBSERVER_PATH)) | _attribute_names(_parse(CLASSIFIER_PATH))
  assert "actuators" not in attrs, "observer must never read/write carControl.actuators"
  assert "actuatorsOutput" in attrs, "sanity check: hMeasured should come from carOutput.actuatorsOutput"


def test_pubmaster_publishes_only_its_own_two_services():
  tree = _parse(OBSERVER_PATH)
  pubmaster_args = None
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
      if name == "PubMaster":
        pubmaster_args = [elt.value for elt in node.args[0].elts]
  assert pubmaster_args == ["rackEffortFrame", "rackEffortSnapshot"]


def test_send_calls_only_target_its_own_services():
  tree = _parse(OBSERVER_PATH)
  allowed = {"rackEffortFrame", "rackEffortSnapshot"}
  for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "send":
      first = node.args[0]
      if isinstance(first, ast.Constant):
        assert first.value in allowed, f"pm.send() targets an unexpected service: {first.value}"


def test_no_submaster_subscription_to_selfdrive_state_or_carcontrol_write_topics():
  """carControl is subscribed (read-only, for latActive -- extract.py's own gate), but
  nothing that writes actuators (e.g. sendcan) is ever a SubMaster topic here."""
  tree = _parse(OBSERVER_PATH)
  forbidden = {"sendcan", "carControllerParams"}
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
      if name == "SubMaster":
        services = [elt.value for elt in node.args[0].elts]
        assert forbidden.isdisjoint(services), f"observer subscribes to a write-path service: {services}"
