# Statistical test methods for SLANT-TICV

SLANT-TICV, as an improved version of SLANT, does not come with any QA files for checking the rationality of the output results by itself. After running SLANT-TICV on a raw dataset, we usually lack the method to assess the rationality of the results and identify the failed files during the operation. This program employs a variety of statistical methods to analyze SLANT-TICV and uses data visualization to facilitate understanding.

```
rm -rf logs && mkdir -p logs
nohup python brain_volume.py > logs/out_brain_volumes.log 2>&1 &
nohup python roi_volume.py > logs/out_roi_volumes.log 2>&1 &
nohup python ticv_volume.py > logs/out_ticv_volumes.log 2>&1 &
nohup python asymmetry_index.py > logs/out_ai.log 2>&1 &
```
