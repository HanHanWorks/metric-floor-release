# Reproduce everything that can be reproduced from this repository.
PY ?= python3

.PHONY: all verify tables figures clean

all: verify tables figures

verify:
	$(PY) replication/verify_paper_numbers.py

tables:
	$(PY) src/analysis/trace_tables.py --json results/analysis/trace_tables.json
	$(PY) src/analysis/c19_station_ext.py --json results/analysis/c19.json
	$(PY) src/analysis/c20_holdout.py    --json results/analysis/c20.json
	$(PY) src/analysis/c21_regime.py     --json results/analysis/c21.json
	$(PY) src/analysis/c22_dm.py         --json results/analysis/c22.json

figures:
	@mkdir -p figures
	$(PY) src/figures/make_fig2.py
	$(PY) src/figures/make_fig3.py
	$(PY) src/figures/make_fig4.py
	$(PY) src/figures/make_fig5.py
	$(PY) src/figures/make_fig6.py

clean:
	rm -rf results/analysis
