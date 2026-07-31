#!/usr/bin/env bash
# run_eval.sh — Evaluate base MiniCPM on SumMe and TVSum

echo "=========================================="
echo "  Evaluating and Extracting SumMe"
echo "=========================================="
python ./vslice/test_simple_dpo.py --dataset="summe" > log_summe.txt

echo ""
echo "=========================================="
echo "  Evaluating and Extracting TVSum"
echo "=========================================="
python ./vslice/test_simple_dpo.py --dataset="tvsum" > log_tvsum.txt

echo ""
echo "=========================================="
echo "  All evaluations and extractions complete."
echo "=========================================="
