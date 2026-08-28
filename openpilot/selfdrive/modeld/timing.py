"""Lightweight shared model-action timing contract.

Kept dependency-free so deterministic controller replay does not import the
modeld process and its compiled runtime services.
"""

# BLaTv2 consumes the model-authored scalar action without an additional
# low-pass filter. Controller timing is deliberately defined independently in
# blatv2/reference.py, so changing a modeld filter can never move its action
# point.
LAT_SMOOTH_SECONDS = 0.0
