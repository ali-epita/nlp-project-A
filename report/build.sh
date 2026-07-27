#!/usr/bin/env bash
# Build the report in three formats from report.md: PDF, LaTeX source, and
# Word. Figures are referenced from results/figures, so run the analysis
# scripts first.
set -euo pipefail
cd "$(dirname "$0")"

ENGINE=xelatex
command -v xelatex >/dev/null || ENGINE=tectonic

COMMON=(--from markdown+implicit_figures --resource-path=..:../results/figures)

pandoc report.md "${COMMON[@]}" --pdf-engine="$ENGINE" -o report.pdf
pandoc report.md "${COMMON[@]}" -s -o report.tex
pandoc report.md "${COMMON[@]}" -o report.docx

echo "report/report.pdf report/report.tex report/report.docx"
