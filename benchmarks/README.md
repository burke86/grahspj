# jaxsedfit benchmarks

This directory contains reusable performance benchmarks for `jaxsedfit`.

## PR benchmark

`jaxsedfit_pr_benchmark.py` runs a fixed-redshift Fairall 9 photometric likelihood benchmark and writes JSON plus Markdown output suitable for a pull-request comment. Timing measurements run multiple trials by default and report mean plus standard error.

Local example:

```bash
conda run -n sed python benchmarks/jaxsedfit_pr_benchmark.py run \
  --output-dir /tmp/jaxsedfit-bench \
  --label local \
  --sha local \
  --dsps-ssp-fn ../jaxqsofit/tempdata.h5 \
  --trials 3
```

Compare two benchmark JSON files:

```bash
conda run -n sed python benchmarks/jaxsedfit_pr_benchmark.py compare \
  --baseline-json /tmp/jaxsedfit-bench-base/benchmark.json \
  --candidate-json /tmp/jaxsedfit-bench-head/benchmark.json \
  --output-dir /tmp/jaxsedfit-bench-compare
```

The GitHub workflows in `.github/workflows/pr-benchmarks.yml` and `.github/workflows/publish-pr-benchmarks.yml` run this benchmark for base and PR commits, upload artifacts, and create or update a PR comment containing the pre/post results.
