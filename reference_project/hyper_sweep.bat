ECHO OFF

SET SGDMSTR = sgdm
SET ELUSTR = elu
 
FOR %%y IN (.001, .01, .05, .1) DO FOR %%x IN (0, .1, .5, 1) DO py -3.8 ./main.py --epochs 2 --optmizer %SGDMSTR% --crossvalidation=True --learningrate %%y --momentum %%x

FOR %%y IN (.001, .01, .05, .1) DO FOR %%x IN (0, .1, .5, 1) DO py -3.8 ./main.py --epochs 2 --optmizer %SGDMSTR% --activation %ELUSTR%  --crossvalidation=True --learningrate %%y --momentum %%x

FOR %%y IN (.001, .01, .05, .1) DO FOR %%x IN (0, .1, .5, 1) DO py -3.8 ./main.py --epochs 2 --crossvalidation=True --learningrate %%y --momentum %%x

FOR %%y IN (.001, .01, .05, .1, .5) DO FOR %%x IN (0, .1, .5, 1) DO py -3.8 ./main.py --epochs 2 --crossvalidation=True --activation %ELUSTR% --learningrate %%y --momentum %%x
PAUSE