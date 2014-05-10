__author__ = 'Abuenameh'

import os
import sys
import pyalps
import random
import numpy as np

L = 4

if sys.platform == 'darwin':
    bhdir = '/tmp/BH-DMRG'
elif sys.platform == 'linux2':
    bhdir = '/mnt/BH-DMRG'
filenameprefix = 'BH_MC_'

measurements = ['Energy', 'Density', 'Density^2', 'Local Density', 'Local Density^2', 'Stiffness']

np.random.seed(0)
mu = 2*np.random.random(L*L) - 0.5

parms = []
# for t in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
for t in [0.01]:
    parms.append(
        {
            'TEMP_DIRECTORY': bhdir,
            'MEASURE[Density]': 1,
            'MEASURE[Density^2]': 1,
            'MEASURE[Local Density]': 1,
            'MEASURE_LOCAL[Local Density^2]': "n2",
            # 'LATTICE'        : "square lattice",
            'LATTICE'        : "inhomogeneous square lattice",
            'MODEL'          : "boson Hubbard",
            'T'              : 0.005,
            'L'              : L,
            't'              : t,
            # 'mu'             : 0.5,
            # 'mu'             : "0.5*2*(random()-0.5)",
            'mu'             : 'get(' + str(L) + '*x+y,' + ",".join([str(mui) for mui in mu]) + ')',
            'U'              : 1.0,
            'NONLOCAL'       : 0,
            'Nmax'           : 2,
            # 'USE_1D_STIFFNESS': 1,
            'DISORDERSEED'   : 12345,
            # 'THERMALIZATION' : 100000,
            # 'SWEEPS'         : 1000000
            'THERMALIZATION' : 10000,
            'SWEEPS'         : 500000
        }
    )

os.chdir(bhdir)
input_file = pyalps.writeInputFiles(filenameprefix,parms)

res = pyalps.runApplication('/opt/alps/bin/worm',input_file,Tmin=5)
# data = pyalps.loadMeasurements(pyalps.getResultFiles(prefix=filenameprefix), measurements)
data = pyalps.loadMeasurements(pyalps.getResultFiles(prefix=filenameprefix),['Energy'])
# print data

# for d in data:
#     for s in d:
        # print s.props['observable']

res = pyalps.runApplication('/opt/alps/bin/sparsediag',input_file,Tmin=5)
data2 = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=filenameprefix),['Energy'])

print data[0][0].y
print data2[0][0][0].y
