# Makefile of the PFCA study. Run from the repository root.
#
#   make install      install the package in editable mode with all extras
#   make test         run the test suite
#   make smoke        run the small offline configuration of every phase (a few minutes, two workers)
#   make phase-a      Phase A, synthetic ground truth (full design of Section 6.1)
#   make phase-b      Phase B, public benchmarks (downloads the OpenML datasets)
#   make phase-c      Phase C, questionnaire (needs data/raw/, see data/README.md)
#   make phase-d      Phase D, generate the expert study materials; make phase-d-analyze RESPONSES=<csv> analyzes the returned sheets
#   make sensitivity  sensitivity analyses of Section 7.5
#   make figures      example Pareto front figure and the figures and tables of Table 2 from the saved metric files
#   make lock         regenerate requirements-lock.txt from the current environment
#
# Variables: PYTHON (default python3), RESULTS (default results), OUT (default ., receives figures/ and tables/),
# FIGURES (default figures, directory of the example front figure), N_JOBS (default 4).
#
# Author: Charis Ntakolia, Hellenic Air Force Academy

PYTHON ?= python3
RESULTS ?= results
OUT ?= .
FIGURES ?= figures
N_JOBS ?= 4
CONFIGS := experiments/configs

.PHONY: install test smoke smoke-a smoke-b smoke-c smoke-d smoke-sensitivity smoke-figures \
        phase-a phase-b phase-c phase-d phase-d-analyze sensitivity figures lock

install:
	$(PYTHON) -m pip install -e ".[all]"

test:
	$(PYTHON) -m pytest -q

# ---------------------------------------------------------------------------
# Smoke runs: small configurations that exercise every script offline
# ---------------------------------------------------------------------------
smoke: smoke-a smoke-b smoke-c smoke-d smoke-sensitivity smoke-figures

smoke-a:
	$(PYTHON) experiments/run_phase_a.py --config $(CONFIGS)/phase_a_smoke.yaml --results $(RESULTS) --name phase_a_smoke --n-jobs 2

smoke-b:
	$(PYTHON) experiments/run_phase_b.py --config $(CONFIGS)/phase_b_smoke.yaml --results $(RESULTS) --name phase_b_smoke

smoke-c:
	$(PYTHON) experiments/make_demo_questionnaire.py --n 300 --seed 0
	$(PYTHON) experiments/run_phase_c.py --config $(CONFIGS)/phase_c_smoke.yaml --results $(RESULTS) --name phase_c_smoke

smoke-d:
	$(PYTHON) experiments/run_phase_d.py generate --synthetic --n-instances 3 --n-experts 6 --results $(RESULTS) --name phase_d_smoke --n-resamples 3
	$(PYTHON) experiments/run_phase_d.py analyze --results $(RESULTS) --name phase_d_smoke --simulate

smoke-sensitivity:
	$(PYTHON) experiments/run_sensitivity.py --config $(CONFIGS)/sensitivity_smoke.yaml --results $(RESULTS) --name sensitivity_smoke

smoke-figures:
	$(PYTHON) experiments/example_front.py --n-resamples 4 --n-explain 30 --n-jobs 2 --out $(FIGURES) --results $(RESULTS)
	$(PYTHON) experiments/make_figures.py --results $(RESULTS) --phase-a phase_a_smoke --phase-b phase_b_smoke --phase-c phase_c_smoke --sensitivity sensitivity_smoke --out $(OUT)

# ---------------------------------------------------------------------------
# Full study, phase by phase
# ---------------------------------------------------------------------------
phase-a:
	$(PYTHON) experiments/run_phase_a.py --config $(CONFIGS)/phase_a.yaml --results $(RESULTS) --name phase_a --n-jobs $(N_JOBS)

phase-b:
	$(PYTHON) experiments/run_phase_b.py --config $(CONFIGS)/phase_b.yaml --results $(RESULTS) --name phase_b --n-jobs $(N_JOBS)

phase-c:
	$(PYTHON) experiments/run_phase_c.py --config $(CONFIGS)/phase_c.yaml --results $(RESULTS) --name phase_c --n-jobs $(N_JOBS)

phase-d:
	$(PYTHON) experiments/run_phase_d.py generate --config $(CONFIGS)/phase_d.yaml --results $(RESULTS) --name phase_d --n-jobs $(N_JOBS)
	@echo "Materials written to $(RESULTS)/phase_d. After the response sheets are returned run:"
	@echo "  make phase-d-analyze RESPONSES=$(RESULTS)/phase_d/responses.csv"

RESPONSES ?= $(RESULTS)/phase_d/responses.csv
phase-d-analyze:
	$(PYTHON) experiments/run_phase_d.py analyze --results $(RESULTS) --name phase_d --responses $(RESPONSES)

sensitivity:
	$(PYTHON) experiments/run_sensitivity.py --config $(CONFIGS)/sensitivity.yaml --results $(RESULTS) --name sensitivity --n-jobs $(N_JOBS)

figures:
	$(PYTHON) experiments/example_front.py --n-jobs $(N_JOBS) --out $(FIGURES) --results $(RESULTS)
	$(PYTHON) experiments/make_figures.py --results $(RESULTS) --out $(OUT)

# ---------------------------------------------------------------------------
# Lock file: the scientific stack and its direct dependencies, as installed
# ---------------------------------------------------------------------------
define LOCK_SCRIPT
import importlib.metadata as md
import re
KEY = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'shap', 'pymoo', 'joblib', 'pyyaml', 'matplotlib', 'statsmodels', 'pytest']
def norm(n):
    return re.sub(r'[-_.]+', '-', n).lower()
def direct(name):
    out = []
    for req in md.requires(name) or []:
        if 'extra ==' in req:
            continue
        m = re.match(r'\s*([A-Za-z0-9][A-Za-z0-9._-]*)', req)
        if m:
            out.append(norm(m.group(1)))
    return out
names = set()
for k in KEY:
    try:
        md.version(k)
    except md.PackageNotFoundError:
        continue
    names.add(norm(k))
    names.update(direct(k))
installed = {}
for d in md.distributions():
    installed.setdefault(norm(d.metadata['Name']), (d.metadata['Name'], d.version))
for n in sorted(names):
    if n in installed and n != 'pfca':
        print(installed[n][0] + '==' + installed[n][1])
endef
export LOCK_SCRIPT

lock:
	@echo "# Frozen versions of the scientific stack used for the PFCA study (Section 8.2 of the study design guide)." > requirements-lock.txt
	@echo "# Generated with 'make lock' from pip metadata: the runtime and optional dependencies of the package" >> requirements-lock.txt
	@echo "# (numpy, scipy, pandas, scikit-learn, shap, pymoo, joblib, pyyaml, matplotlib, statsmodels, pytest)" >> requirements-lock.txt
	@echo "# and their direct dependencies, as installed. Use it as a constraint file:" >> requirements-lock.txt
	@echo "#   pip install -c requirements-lock.txt -e \".[all]\"" >> requirements-lock.txt
	@$(PYTHON) -c "$$LOCK_SCRIPT" >> requirements-lock.txt
	@echo "wrote requirements-lock.txt"
