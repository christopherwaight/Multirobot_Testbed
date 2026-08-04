"""
_coords_common.py

Affine world <-> lat/lon coordinate helpers shared by the ocean HFR experiment
and FTLE-overlay scripts. The cluster simulates in "world" coordinates in
[-world_half, world_half]; these functions map to/from geographic lat/lon
around (center_lat, center_lon), using the same parameters stored in a
field's config dict (loaded from config/fields/<name>.yaml).

The scales themselves come from Ocean_HFR.world_scales(). This module used to
recompute them, which let the field evaluator and the plotting/scoring code
drift apart; it now delegates so there is one definition of the map.

Latitude half-extent is roi_half_deg. The longitude scale is derived so that
one world unit is the same distance east as north (isotropic_map, default
true), which makes the world frame a uniform dilation of the local tangent
plane. See Ocean_HFR.py's module docstring for what that buys and what
isotropic_map: false reproduces.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fields.environments.Ocean_HFR import world_scales, world_unit_km  # noqa: F401


def affine(config):
    """Return (center_lat, center_lon, scale_lat, scale_lon) in degrees per world unit."""
    return world_scales(config)


def latlon_to_world(lat, lon, config):
    """Convert geographic (lat, lon) to cluster world (x, y)."""
    clat, clon, scale_lat, scale_lon = world_scales(config)
    return (lon - clon) / scale_lon, (lat - clat) / scale_lat


def world_to_latlon(x, y, config):
    """Convert cluster world (x, y) to geographic (lat, lon)."""
    clat, clon, scale_lat, scale_lon = world_scales(config)
    return clat + y * scale_lat, clon + x * scale_lon
