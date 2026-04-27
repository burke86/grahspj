# grahspj benchmarks

This directory contains reusable performance benchmarks for `grahspj`.

## PR benchmark

`grahspj_pr_benchmark.py` runs a fixed-redshift Fairall 9 photometric likelihood benchmark and writes JSON plus Markdown output suitable for a pull-request comment.

Local example:

```bash
conda run -n sed python benchmarks/grahspj_pr_benchmark.py run \
  --output-dir /tmp/grahspj-bench \
  --label local \
  --sha local \
  --dsps-ssp-fn ../jaxqsofit/tempdata.h5
```

Compare two benchmark JSON files:

```bash
conda run -n sed python benchmarks/grahspj_pr_benchmark.py compare \
  --baseline-json /tmp/grahspj-bench-base/benchmark.json \
  --candidate-json /tmp/grahspj-bench-head/benchmark.json \
  --output-dir /tmp/grahspj-bench-compare
```

The GitHub workflows in `.github/workflows/pr-benchmarks.yml` and `.github/workflows/publish-pr-benchmarks.yml` run this benchmark for base and PR commits, upload artifacts, and create or update a PR comment containing the pre/post results.
