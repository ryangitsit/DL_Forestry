ECHO OFF

FOR %%z IN (SGD,RMS,adam) DO FOR %%y IN (.001, .01, .1) DO FOR %%x IN (0, .1, 1) DO FOR %%a IN (True, False) DO py -3.8 ./main.py --epochs 2 --optimizer %%z  --learningrate %%y --momentum %%x --augmentation %%a

PAUSE