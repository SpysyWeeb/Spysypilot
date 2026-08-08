from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BuildDescriptor,
  BuildDescriptorRegistry,
)


REGISTRY_PATH = (
  Path(__file__).parents[1]
  / "lib"
  / "blatv2"
  / "historical_build_descriptors.json"
)
REVIEWED_REGISTRY_SHA256 = (
  "297a8d72fa7e6c3a61e981fd875c919be043601f505f8c46a749403c24ca7214"
)
FIRST_DESCRIPTOR_SHA256 = (
  "8ce8026f4ff30b1036206046dcd96aa3b27a47f61f21673a2629154379769684"
)
INTENDED_ROOT_COMMITS = {
  "02bf07c412b4ae92889c7a977fec61328a300c66",
  "04a9f12d75c27ee349cd883bbd4ec68c0cc99413",
  "080f2a55a07484a6f71cfe9706c1fdaea5dca5d8",
  "0cf7aeda81d66ca340f5f80ec581745066818473",
  "1021699bac528ba4ce39db23990c4d2e7867d4ba",
  "22b915a7ab7a1d2963c12db9cc6a48bda3be708f",
  "2447667ea36160b7706b8ab919bc2d4e71b54f56",
  "29385dd8d001ebb8452ba1fc9bbb66858fdbd778",
  "34357419f38b84e2de90bc9cbfdfb5704a282627",
  "36bcc1be838138d39e90a70e8ba5277dc87e04a3",
  "3849a2f72e8fe1902dc4b91c4a3b98384295103a",
  "3b41587559d7822986565dcd13904cfb0c3aae2e",
  "49d7f41ae428464ee9cb95a3e381f45f0865ba9d",
  "624d4c7677947cedf516d2bfad88591795975557",
  "73e1d56cb4fec4a819b1d1a925e70a124114684b",
  "75caa41962e0f41351e30faef2652935a0af6a92",
  "7a1ddfc50c3dbb1a8837f95929474156d8c2ee46",
  "8341af9232c3ff1b0f99163b8a1b5f781d0fd47c",
  "86c85015d75a5272a4b78127ce28fc596708a968",
  "8cc8a31d22fc54ca219f06d77e3dcba7b080c228",
  "9338f5be95d0b6c68bb326945e18c4e21e4b8147",
  "95ee7b64dfca375280519476f26f5500f3ae40d5",
  "9b9e2fa99ca915edd48cfe2767bb99f77240b895",
  "9aabf0fadb47c8ccfb9918c0d70f002ca7be77ef",
  "abc04f788f4a3624db5e99dec5f16fcf779c4757",
  "af8c4fe3523fa455aa18a16da8295bff054c1875",
  "b1006825028dc268b1334405626690d73d56fa0c",
  "b1328fe898fa25e4e8584fd9b3caf80deefb81e8",
  "b8bea34ddac98ab40dcf5d2eb1aa4dda3b120a8c",
  "d0002f5286be81e022f5b12b831d9f45c829bb4e",
  "e14723136c5202316770ed3e5b09f5bb2ee39c28",
  "e410f73c6c30e43fe89ae45fb2d27410c6ea7a8a",
  "e45707824f09626e78b54b733e0f8e30ee2ca3bd",
  "ed0f289d46a5794657ceffec6b761b6ab03a9aa7",
  "eed5d8aa664e0fe49c438f69b3a593b4145fa470",
  "efd9336c8ced71040fb940bad322dac8f7bcc3a8",
  "f4f9ec82863befde6bc8448a9e142daf8a935ca0",
  "f826d0d5386a83bd310e3b7c8641637fecbd2046",
  "fbe08814ca352705f5aa313f2f67e2388f6de6f0",
  "fdd5560008aacd1da6960044f09c9b509cd3f463",
  "ff337860920d59db63e24679340a532cb980b732",
  "ff84244da56fcc9050eb04d01d08f6977f35d408",
}


def canonical_bytes(payload: object) -> bytes:
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode("utf-8")


def load_reviewed_registry(path: Path) -> BuildDescriptorRegistry:
  """Load only the exact source-reviewed physical-compatibility registry."""
  registry = BuildDescriptorRegistry.from_json_file(path)
  if registry.identity_sha256 != REVIEWED_REGISTRY_SHA256:
    raise ValueError("historical build registry review identity mismatch")
  return registry


def test_checked_in_registry_is_canonical_complete_and_limits_only() -> None:
  encoded = REGISTRY_PATH.read_bytes()
  payload = json.loads(encoded)
  assert encoded == canonical_bytes(payload)

  descriptors = payload["descriptors"]
  roots = [item["superproject_commit"] for item in descriptors]
  assert len(roots) == len(INTENDED_ROOT_COMMITS)
  assert len(roots) == len(set(roots))
  assert set(roots) == INTENDED_ROOT_COMMITS
  assert roots == sorted(roots)
  assert {
    item["supported_vehicle_identity"]
    for item in descriptors
  } == {"HYUNDAI_PALISADE"}
  assert {
    (
      item["steer_max"],
      item["steer_delta_up"],
      item["steer_delta_down"],
      item["steer_step"],
      item["driver_allowance"],
      item["driver_multiplier"],
      item["driver_factor"],
      item["production_envelope_verified"],
      item["rack_rate_resolution_deg_s"],
    )
    for item in descriptors
  } == {(409, 4, 7, 1, 50, 2, 1, True, 4.0)}
  assert load_reviewed_registry(REGISTRY_PATH).identity_sha256 == (
    REVIEWED_REGISTRY_SHA256
  )


@pytest.mark.parametrize(
  ("root_commit", "opendbc_commit"),
  (
    (
      "8cc8a31d22fc54ca219f06d77e3dcba7b080c228",
      "ab40b765445d1d18750b58ca6524b16ebe219b6b",
    ),
    (
      "29385dd8d001ebb8452ba1fc9bbb66858fdbd778",
      "ab40b765445d1d18750b58ca6524b16ebe219b6b",
    ),
    (
      "624d4c7677947cedf516d2bfad88591795975557",
      "ab40b765445d1d18750b58ca6524b16ebe219b6b",
    ),
    (
      "1021699bac528ba4ce39db23990c4d2e7867d4ba",
      "68fda8e06e648fd23e2cdac6a5d04ef3df67f29b",
    ),
  ),
)
def test_route_builds_have_exact_reviewed_descriptors(
  root_commit: str,
  opendbc_commit: str,
) -> None:
  descriptor = load_reviewed_registry(REGISTRY_PATH).resolve(root_commit)
  assert descriptor is not None
  assert descriptor.to_dict() == {
    "driver_allowance": 50,
    "driver_factor": 1,
    "driver_multiplier": 2,
    "log_schema_blob": "d40096ff46dc7d1b0dec3698e3e9c77a63b3fb72",
    "opendbc_commit": opendbc_commit,
    "panda_commit": "7f245a890f7bc00712ca4ebf903190a084c7f86b",
    "production_envelope_verified": True,
    "rack_rate_resolution_deg_s": 4.0,
    "steer_delta_down": 7,
    "steer_delta_up": 4,
    "steer_max": 409,
    "steer_step": 1,
    "superproject_commit": root_commit,
    "supported_vehicle_identity": "HYUNDAI_PALISADE",
  }


def test_duplicate_root_is_rejected(tmp_path: Path) -> None:
  payload = json.loads(REGISTRY_PATH.read_bytes())
  payload["descriptors"].insert(1, dict(payload["descriptors"][0]))
  duplicate_path = tmp_path / "duplicate.json"
  duplicate_path.write_bytes(canonical_bytes(payload))
  with pytest.raises(ValueError, match="duplicate superproject"):
    BuildDescriptorRegistry.from_json_file(duplicate_path)


def test_unknown_build_is_not_resolved() -> None:
  registry = load_reviewed_registry(REGISTRY_PATH)
  assert registry.resolve("0" * 40) is None


@pytest.mark.parametrize(
  ("field", "replacement"),
  [
    ("opendbc_commit", "0" * 40),
    ("panda_commit", "1" * 40),
    ("log_schema_blob", "2" * 40),
    ("supported_vehicle_identity", "HYUNDAI_SONATA"),
    ("steer_max", 408),
    ("steer_delta_up", 3),
    ("steer_delta_down", 6),
    ("steer_step", 2),
    ("driver_allowance", 49),
    ("driver_multiplier", 3),
    ("driver_factor", 2),
    ("rack_rate_resolution_deg_s", 3.0),
  ],
)
def test_review_identity_rejects_semantic_tampering(
  tmp_path: Path,
  field: str,
  replacement: object,
) -> None:
  payload = json.loads(REGISTRY_PATH.read_bytes())
  payload["descriptors"][0][field] = replacement
  tampered_path = tmp_path / f"tampered-{field}.json"
  tampered_path.write_bytes(canonical_bytes(payload))

  # Values remain structurally well-formed on purpose: the pinned review
  # identity, not permissive parsing, is the semantic integrity boundary.
  with pytest.raises(ValueError, match="review identity mismatch"):
    load_reviewed_registry(tampered_path)


def test_descriptor_and_registry_identities_are_deterministic() -> None:
  payload = json.loads(REGISTRY_PATH.read_bytes())
  first_descriptor_sha256 = hashlib.sha256(
    canonical_bytes(payload["descriptors"][0]),
  ).hexdigest()
  first = load_reviewed_registry(REGISTRY_PATH)
  second = load_reviewed_registry(REGISTRY_PATH)

  assert first_descriptor_sha256 == FIRST_DESCRIPTOR_SHA256
  assert hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest() == (
    REVIEWED_REGISTRY_SHA256
  )
  assert first.identity_sha256 == second.identity_sha256
  assert first.identity_sha256 == REVIEWED_REGISTRY_SHA256


def test_direct_descriptor_construction_rejects_unverified_envelope() -> None:
  payload = json.loads(REGISTRY_PATH.read_bytes())["descriptors"][0]
  payload["production_envelope_verified"] = False
  with pytest.raises(ValueError, match="provenance/envelope"):
    BuildDescriptor(**payload)

  payload["production_envelope_verified"] = True
  payload["rack_rate_resolution_deg_s"] = 0.0
  with pytest.raises(ValueError, match="provenance/envelope"):
    BuildDescriptor(**payload)
