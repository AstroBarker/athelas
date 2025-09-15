#!/bin/bash

: ${CFM:=clang-format}
: ${VERBOSE:=0}

if ! command -v ${CFM} &> /dev/null; then
    >&2 echo "[format.sh] Error: No clang format found! Looked for ${CFM}"
    exit 1
else
    CFM=$(command -v ${CFM})
    echo "[format.sh] Clang format found: ${CFM}"
fi

# clang format major version
TARGET_CF_VRSN=20.1.0
CF_VRSN=$(${CFM} --version)
echo "[format.sh] Note we assume clang format version ${TARGET_CF_VRSN}."
echo "You are using ${CF_VRSN}."
echo "If these differ, results may not be stable."

echo "[format.sh] Formatting..."
REPO=$(git rev-parse --show-toplevel)
for f in $(git grep --untracked -ail ';' -- ':/*.hpp' ':/*.cpp'); do
    if [ ${VERBOSE} -ge 1 ]; then
       echo ${f}
    fi
    ${CFM} -i ${f}
done

# --- Check for ruff ---
if command -v ruff &> /dev/null; then
    echo "[format.sh] Ruff found: $(command -v ruff)"

    PY_FILES=$(git ls-files "scripts/python/*.py")
    if [ -z "${PY_FILES}" ]; then
        echo "[format.sh] No tracked Python files in scripts/python to lint."
    else
        echo "[format.sh] Running ruff on Python files..."
        ruff check --fix ${PY_FILES}
        ruff format ${PY_FILES}
        echo "[format.sh] Ruff linting complete."
    fi
else
    echo "[format.sh] Ruff not found. Skipping Python linting."
fi

echo "[format.sh] ...Done"
