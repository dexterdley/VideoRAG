#!/bin/bash

python ./VideoRAG-algorithm/measure_t3.py
python ./VideoRAG-algorithm/measure_t3.py --mode="bitemporal"

# Overall Expected Calibration Error (ECE): 0.0225
# Average ECE:, 4.5198%
# Overall Expected Calibration Error (ECE): 0.0230
# Average ECE:, 3.7990%