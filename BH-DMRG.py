__author__ = 'Abuenameh'

import sys
import pyalps
import numpy as np
import itertools
import concurrent.futures
import mathematica as m
import datetime
from progressbar import *

numthreads = 6

L = 5
sweeps = 4
maxstates = 100
Nmax = 7

if len(sys.argv) < 3:
    print 'Insufficient number of command line arguments.'
    quit(1)

delta = float(sys.argv[2])
if delta == 0:
    lattice = "open chain lattice"
else:
    lattice = "inhomogeneous chain lattice"

parmsbase = {
    'LATTICE' : lattice,
    'MODEL' : "boson Hubbard",
    'CONSERVED_QUANTUMNUMBERS' : 'N',
    'SWEEPS' : sweeps,
    'NUMBER_EIGENVALUES' : 1,
    'L' : L,
    'MAXSTATES' : maxstates,
    'Nmax' : Nmax,
    'U' : 1
}

if delta > 0:
    parmsbase['delta'] = delta
    parmsbase['mu'] = 'delta*2*(random() - 0.5)'

def rundmrg(i, N, t):
    filenameprefix = 'files/BH_' + str(i)
    parms = [dict(parmsbase.items() + { 'N_total' : N, 't' : t }.items())]
    input_file = pyalps.writeInputFiles(filenameprefix, parms)
    res = pyalps.runApplication('/opt/alps/bin/dmrg', input_file, writexml=True)
    data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=filenameprefix))
    for s in data[0]:
        if s.props['observable'] == 'Energy':
            E0 = s.y[0]
    return [N, t, E0]

# Ns = [1,2,3]#range(1, 2*L, 1)
# ts = [0.1]
Ns = range(1, 2*L, 1)
ts = np.linspace(0.01, 0.3, 5).tolist()

futures = []

E0res = np.zeros([len(Ns), len(ts)])
E0res.fill(np.NaN)

# widgets = [Percentage()]
pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(Ns)*len(ts)).start()

start = datetime.datetime.now()

with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
    for i, Nt in enumerate(itertools.product(Ns, ts)):
        futures.append(executor.submit(rundmrg, i, Nt[0], Nt[1]))
    for future in concurrent.futures.as_completed(futures):
        try:
            res = future.result()
        except Exception as exc:
            print exc
            pass
        else:
            E0res[Ns.index(res[0])][ts.index(res[1])] = res[2]
        pbar.update(pbar.currval+1)

print

end = datetime.datetime.now()
print end - start

resi = sys.argv[1]
resfile = '/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/BH-DMRG/res.' + str(resi) + '.txt'
# resfile = '/Users/Abuenameh/Documents/Simulation Results/BH-DMRG/res.' + str(resi) + '.txt'
resf = open(resfile, 'w')
res = 'Lres[{0}]={1};\nsweeps[{0}]={2};\nmaxstates[{0}]={3};\nNmax[{0}]={4};\nNres[{0}]={5};\ntres[{0}]={6};\nE0res[{0}]={7};\nruntime[{0}]=\"{8}\";\n'.format(resi, L, sweeps, maxstates, Nmax, m.mathformat(Ns), m.mathformat(ts), m.mathformat(E0res), end-start)
resf.write(res)


