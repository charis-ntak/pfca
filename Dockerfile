# Container image of the PFCA study (Section 8.2 of the study design guide).
# Builds from the official slim Python 3.11 image, installs the package with every extra
# under the frozen versions of requirements-lock.txt, and runs the test suite as a build check.
#
#   docker build -t pfca .
#   docker run --rm pfca                                   # runs the tests
#   docker run --rm -v "$PWD/results:/opt/pfca/results" pfca make smoke
#   docker run --rm -v "$PWD/results:/opt/pfca/results" pfca \
#       python experiments/run_phase_a.py --config experiments/configs/phase_a.yaml --results results
#
# Author: Charis Ntakolia, Hellenic Air Force Academy

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# make is used by the Makefile targets; nothing else beyond the wheels of the Python stack is required
RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pfca

COPY pyproject.toml README.md LICENSE CITATION.cff requirements.txt requirements-lock.txt Makefile ./
COPY src ./src
COPY tests ./tests
COPY experiments ./experiments
COPY docs ./docs
COPY data ./data

RUN python -m pip install --upgrade pip \
    && python -m pip install -c requirements-lock.txt -e ".[all]"

# build check: the unit tests and the numerical checks of the properties of Section 5
RUN python -m pytest -q -p no:cacheprovider

RUN mkdir -p results figures tables

CMD ["python", "-m", "pytest", "-q"]
