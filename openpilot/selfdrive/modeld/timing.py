"""Lightweight shared model-action timing contract.

Kept dependency-free so deterministic controller replay does not import the
modeld process and its compiled runtime services.
"""

LAT_SMOOTH_SECONDS = 0.0
