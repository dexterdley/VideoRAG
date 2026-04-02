#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python ./VideoRAG-algorithm/measure_t3.py
CUDA_VISIBLE_DEVICES=7 python ./VideoRAG-algorithm/measure_t3.py --mode="bitemporal"

#Total Frames Evaluated: 5946
#Overall Expected Calibration Error (ECE): 0.0233
#Average ECE:, 4.4218%

# Overall Expected Calibration Error (ECE): 0.0230
# Average ECE:, 3.7990%