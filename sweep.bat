ECHO OFF

FOR %%z IN (RMS,adam) DO FOR %%y IN (.001, .01, .1) DO FOR %%x IN (0) DO FOR %%a IN (False) DO py -3.8 ./main.py --epochs 1 --optimizer %%z  --learningrate %%y --momentum %%x --augmentation %%a

PAUSE