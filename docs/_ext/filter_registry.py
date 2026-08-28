"""Render the complete packaged filter registry in the Sphinx documentation."""

from __future__ import annotations

from collections import defaultdict
from html import escape

from docutils import nodes
from docutils.parsers.rst import Directive


_FAMILY_LABELS = {
    "2mass": "2MASS",
    "cfht": "CFHT",
    "ctio": "CTIO",
    "euclid": "Euclid",
    "galex": "GALEX",
    "generic": "Generic photometric systems",
    "herschel": "Herschel",
    "hst": "Hubble Space Telescope (HST)",
    "jwst": "James Webb Space Telescope (JWST)",
    "kpno": "KPNO",
    "lasilla": "La Silla Observatory",
    "noao": "NOAO",
    "panstarrs": "Pan-STARRS",
    "paranal": "Paranal Observatory",
    "roman": "Nancy Grace Roman Space Telescope",
    "rubin": "Vera C. Rubin Observatory",
    "sloan": "Sloan Digital Sky Survey (SDSS)",
    "spitzer": "Spitzer Space Telescope",
    "subaru": "Subaru Telescope",
    "ukirt": "UKIRT",
    "wise": "WISE",
}


def _display_instrument(parts: list[str]) -> str:
    if len(parts) <= 2:
        return "—"
    return " / ".join(part.replace("_", " ").upper() for part in parts[1:-1])


def _registry_html() -> str:
    from jaxsedfit.filters import (
        FILTER_NAME_ALIASES,
        load_filter_curve,
        vendored_filter_registry,
    )

    aliases = defaultdict(list)
    for alias, canonical in FILTER_NAME_ALIASES.items():
        aliases[canonical].append(alias)

    families = defaultdict(list)
    for canonical in vendored_filter_registry():
        parts = canonical.split(".")
        curve = load_filter_curve(canonical)
        families[parts[0]].append(
            (
                float(curve.effective_wavelength),
                _display_instrument(parts),
                parts[-1],
                canonical,
                tuple(sorted(aliases.get(canonical, ()))),
            )
        )

    nav = "".join(
        f'<a href="#filters-{escape(family)}">{escape(_FAMILY_LABELS.get(family, family.upper()))}</a>'
        for family in sorted(families)
    )
    sections = []
    for family in sorted(families):
        rows = []
        for pivot, instrument, filter_name, canonical, public_aliases in sorted(
            families[family]
        ):
            alias_html = (
                ", ".join(f"<code>{escape(alias)}</code>" for alias in public_aliases)
                if public_aliases
                else "—"
            )
            rows.append(
                "<tr>"
                f"<td>{escape(instrument)}</td>"
                f"<td><code>{escape(filter_name)}</code></td>"
                f"<td><code>{escape(canonical)}</code></td>"
                f'<td class="numeric">{pivot:,.1f}</td>'
                f"<td>{alias_html}</td>"
                "</tr>"
            )
        label = _FAMILY_LABELS.get(family, family.upper())
        sections.append(
            f'<section class="filter-family" id="filters-{escape(family)}">'
            f"<h2>{escape(label)} <span>({len(rows)} filters)</span></h2>"
            '<div class="filter-table-scroll"><table class="filter-reference-table">'
            "<thead><tr><th>Instrument / system</th><th>Filter</th>"
            "<th>Canonical name</th><th>Pivot wavelength [Å]</th>"
            "<th>Accepted aliases</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></section>"
        )

    return (
        '<div class="filter-reference">'
        f'<p class="filter-count"><strong>{sum(map(len, families.values()))}</strong> canonical filters '
        f"across <strong>{len(families)}</strong> telescope and survey families.</p>"
        f'<nav class="filter-family-nav">{nav}</nav>{"".join(sections)}</div>'
    )


class FilterRegistryDirective(Directive):
    """Insert all packaged filters grouped by telescope and instrument."""

    has_content = False

    def run(self):
        return [nodes.raw("", _registry_html(), format="html")]


def setup(app):
    """Register the filter-registry directive with Sphinx."""
    app.add_directive("filter-registry", FilterRegistryDirective)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
