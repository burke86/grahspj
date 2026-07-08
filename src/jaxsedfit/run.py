from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from .config import FitConfig, fit_config_from_mapping
from .core import JAXSEDFit


def _load_config(path: str) -> FitConfig:
    """Load a fit configuration from a JSON file or Python module.

    Parameters
    ----------
    path : object
        path value.
    """
    cfg_path = Path(path)
    if cfg_path.suffix == ".json":
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return fit_config_from_mapping(json.load(fh))
    spec = importlib.util.spec_from_file_location("jaxsedfit_user_config", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "CONFIG"):
        config_obj = module.CONFIG
    elif hasattr(module, "build_config"):
        config_obj = module.build_config()
    else:
        raise AttributeError("Config file must define CONFIG or build_config().")
    if isinstance(config_obj, FitConfig):
        return config_obj
    if isinstance(config_obj, dict):
        return fit_config_from_mapping(config_obj)
    raise TypeError("Unsupported config object type.")


def main(argv: list[str] | None = None) -> int:
    """Run the jaxsedfit single-object CLI entrypoint.

    Parameters
    ----------
    argv : object
        argv value.
    """
    parser = argparse.ArgumentParser(description="Run a single-source jaxsedfit fit.")
    parser.add_argument("config", help="Path to a Python or JSON fit config.")
    parser.add_argument("--method", choices=("map", "nuts"), default="map")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    fitter = JAXSEDFit(config)
    if args.method == "map":
        fitter.fit_map()
    else:
        fitter.fit_nuts()
    out = fitter.save(args.output_dir)
    print(json.dumps(fitter.summary(), indent=2, default=str))
    print(f"Saved posterior bundle to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
