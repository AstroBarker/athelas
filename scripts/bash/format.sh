#!/bin/bash

: ${CFM:=clang-format}
: ${PFM:=black}
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

# format python files
if ! command -v ${PFM} &> /dev/null; then
    >&2 echo "[format.sh] Error: No version of ruff found! Looked for ${PFM}"
    exit 1
else
    PFM=$(command -v ${PFM})
    echo "ruff Python formatter found: ${PFM}"
    echo "ruff version: $(${PFM} --version)"
fi

echo "Formatting Python files..."
REPO=$(git rev-parse --show-toplevel)
for f in $(git grep --untracked -ail res -- :/*.py); do
    if [ ${VERBOSE} -ge 1 ]; then
       echo ${f}
    fi
    ${PFM} -q ${REPO}/${f}
done
echo "[format.sh] ...Done"
