"""
_coords_common.py

Affine world <-> lat/lon coordinate helpers shared by the ocean HFR experiment
and FTLE-overlay scripts. The cluster simulates in "world" coordinates in
[-world_half, world_half]; these functions map to/from geographic lat/lon
within +/- roi_half_deg of (center_lat, center_lon), using the same
(center_lat, center_lon, roi_half_deg, world_half) parameters stored in a
field's config dict (loaded from config/fields/<name>.yaml).
"""


def affine(config):
    """Return (center_lat, center_lon, scale) where world_coord * scale = degrees."""
    clat = config.get("center_lat", 34.2)
    clon = config.get("center_lon", -120.4)
    scale = config.get("roi_half_deg", 0.3) / config.get("world_half", 0.65)
    return clat, clon, scale


def latlon_to_world(lat, lon, config):
    """Convert geographic (lat, lon) to cluster world (x, y)."""
    clat, clon, scale = affine(config)
    return (lon - clon) / scale, (lat - clat) / scale


def world_to_latlon(x, y, config):
    """Convert cluster world (x, y) to geographic (lat, lon)."""
    clat, clon, scale = affine(config)
    return clat + y * scale, clon + x * scale
