# Third-party Notices

## GRAHSP / pcigale

Portions of `grahspj` are derived from or closely based on code and data from `GRAHSP` / `pcigale`.

- Upstream project: `GRAHSP`
- Upstream license: `CeCILL v2`
- Local copy of license text: [CeCILL-v2.txt](CeCILL-v2.txt)

Relevant upstream source files include, among others:

- `pcigale/creation_modules/activate.py`
- `pcigale/creation_modules/activategtorus.py`
- `pcigale/creation_modules/activatelines.py`
- `pcigale/creation_modules/biattenuation.py`
- `pcigale/creation_modules/redshifting.py`
- `pcigale/creation_modules/galdale2014.py`

This repository contains JAX/NumPyro ports and modifications of selected model behavior, plus redistributed resource files used by the current supported subset.

## Vendored resource files

The following bundled resource categories in `src/grahspj/resources/` originate from upstream `GRAHSP` resources or associated template bundles used by `GRAHSP`:

- `resources/filters/filter_registry.txt`
- `resources/filters/*`
- `resources/templates/Fe_d11-m20-20.5.txt`
- `resources/templates/emission_line_table.formatted`
- `resources/templates/dale2014/*`

See the README files in those resource directories for per-directory provenance notes.
