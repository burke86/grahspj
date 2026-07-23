"""Shared NumPyro geometry helpers for native and embedded spectral fits."""

from __future__ import annotations

from .spectral_model import (
    _line_amplitude_site,
    _ordered_width_site,
)


def line_complex_dense_mass_blocks(tied_line_meta, *, standardized_amplitudes):
    """Return compact dense blocks for line complexes and ordered widths.

    Each complex gets a local amplitude/centroid block. Ordered-width
    coordinates join the block for their own complex, while the global
    broad-width hyperparameters remain diagonal. Keeping ordered complexes
    separate avoids estimating spurious cross-complex covariance and was
    measurably more efficient than one shared width block.
    """
    blocks = []
    width_complexes = list(
        tied_line_meta.get("broad_width_order_complex_indices", [])
    )
    width_labels = list(
        tied_line_meta.get("broad_width_order_site_labels", [])
    )
    centroid_hierarchies = list(
        tied_line_meta.get("broad_centroid_hierarchy_groups", [])
    )
    ordered_owner_indices = {int(index) for index in width_complexes}
    ordered_complex_sites = {}
    for complex_group in tied_line_meta.get("amp_complex_groups", []):
        complex_index = int(complex_group["complex_index"])
        complex_label = str(
            complex_group.get("site_label", f"complex_{complex_index}")
        )
        sites = [
            _line_amplitude_site(
                complex_label, standardized=standardized_amplitudes
            )
        ]
        for hierarchy_index, hierarchy in enumerate(centroid_hierarchies):
            if int(hierarchy.get("complex_index", -1)) == complex_index:
                sites.extend(
                    [
                        f"line_broad_center_{hierarchy_index}_std",
                        f"line_broad_relative_offsets_{hierarchy_index}_std",
                    ]
                )
        if complex_index in ordered_owner_indices:
            ordered_complex_sites[complex_index] = sites
        else:
            blocks.append(tuple(sites))

    ordered_width_sites = {}
    for order_index, owner_index in enumerate(width_complexes):
        owner_index = int(owner_index)
        order_label = (
            str(width_labels[order_index])
            if order_index < len(width_labels)
            else str(order_index)
        )
        ordered_width_sites.setdefault(owner_index, []).append(
            _ordered_width_site(order_label, standardized=True)
        )
    for owner_index in dict.fromkeys(int(index) for index in width_complexes):
        sites = tuple(ordered_complex_sites.get(owner_index, ())) + tuple(
            ordered_width_sites.get(owner_index, ())
        )
        if sites:
            blocks.append(sites)
    return blocks
