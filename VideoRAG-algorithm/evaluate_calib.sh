#!/bin/bash

python ./VideoRAG-algorithm/measure_t3.py
python ./VideoRAG-algorithm/measure_t3.py --mode="bitemporal"

# Overall Expected Calibration Error (ECE): 0.0233                                
# Average ECE:, 4.4218%
# Overall Expected Calibration Error (ECE): 0.0230
# Average ECE:, 3.7990%