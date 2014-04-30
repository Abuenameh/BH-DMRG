from __future__ import print_function
__author__ = 'Abuenameh'

import sys
import pyalps
import numpy as np
import itertools
import concurrent.futures
import mathematica as m
import datetime
import gtk
import gobject
import threading
import os
from mathematica import mathformat
from switch import switch
from speed import *

numthreads = 4

L = 30
sweeps = 4
maxstates = 100
nmax = 7

if len(sys.argv) < 3:
    print('Insufficient number of command line arguments.')
    quit(1)

delta = float(sys.argv[2])
if delta == 0:
    lattice = "open chain lattice"
else:
    lattice = "inhomogeneous chain lattice"

bhdir = '/mnt/BH-DMRG'
# bhdir = '/tmp/BH-DMRG'
filenameprefix = 'BH_'

parmsbase = {
    'TEMP_DIRECTORY' : bhdir,
    'LATTICE' : lattice,
    'MODEL' : "boson Hubbard",
    'CONSERVED_QUANTUMNUMBERS' : 'N',
    'SWEEPS' : sweeps,
    'NUMBER_EIGENVALUES' : 1,
    'L' : L,
    'MAXSTATES' : maxstates,
    'Nmax' : nmax,
    'U' : 1#,
    # 'MEASURE_LOCAL[Local density]' : "n",
    # 'MEASURE_LOCAL[Local density squared]' : "n2",
    # 'MEASURE_CORRELATIONS[Correlation function]' : "bdag:b"
}

if delta > 0:
    parmsbase['delta'] = delta
    parmsbase['mu'] = 'delta*2*(random() - 0.5)'

def rundmrg(i, t, N, it, iN):
    parms = [dict(parmsbase.items() + { 'N_total' : N, 't' : t, 'it' : it, 'iN' : iN }.items())]
    input_file = pyalps.writeInputFiles(filenameprefix + str(i), parms)
    pyalps.runApplication('/opt/alps/bin/dmrg', input_file, writexml=True)


def runmain():
    ts = np.linspace(0.01, 0.3, 1).tolist()
    Ns = range(1, 2*L+1, 2*L)

    dims = [ len(ts), len(Ns) ]
    ndims = dims + [ L ]
    Cdims = dims + [ L, L ]

    Trunc = np.zeros(dims)

    E0res = np.zeros(dims)
    nres = np.zeros(ndims)
    n2res = np.zeros(ndims)
    Cres = np.zeros(Cdims)

    E0res.fill(np.NaN)
    nres.fill(np.NaN)
    n2res.fill(np.NaN)
    Cres.fill(np.NaN)

    start = datetime.datetime.now()

    with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
        futures = [executor.submit(rundmrg, i, tN[0][0], tN[0][1], tN[1][0], tN[1][1]) for i, tN in enumerate(zip(itertools.product(ts, Ns), itertools.product(range(0, len(ts)), range(0, len(Ns)))))]
        for future in gprogress(concurrent.futures.as_completed(futures), size=len(futures)):
            pass

    data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=filenameprefix))
    for d in data:
        for s in d:
            it = int(s.props['it'])
            iN = int(s.props['iN'])
            for case in switch(s.props['observable']):
                if case('Truncation error'):
                    Trunc[it][iN] = s.y[0]
                    break
                if case('Energy'):
                    E0res[it][iN] = s.y[0]
                    break
                if case('Local density'):
                    nres[it][iN] = s.y[0]
                    break
                if case('Local density squared'):
                    n2res[it][iN] = s.y[0]
                    break
                if case('Correlation function'):
                    Cres[it][iN] = np.split(s.y[0], L)
                    break
            # if s.props['observable'] == 'Energy':
            #     E0res[int(s.props['it'])][int(s.props['iN'])] = s.y[0]

    end = datetime.datetime.now()
    # print(end - start)

    resi = sys.argv[1]
    resfile = '/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/BH-DMRG/res.' + str(resi) + '.txt'
    # resfile = '/Users/Abuenameh/Documents/Simulation Results/BH-DMRG/res.' + str(resi) + '.txt'
    resf = open(resfile, 'w')
    # res = 'Lres[{0}]={1};\nsweeps[{0}]={2};\nmaxstates[{0}]={3};\nNmax[{0}]={4};\nNres[{0}]={5};\ntres[{0}]={6};\nE0res[{0}]={7};\nruntime[{0}]=\"{8}\";\n'.format(resi, L, sweeps, maxstates, Nmax, m.mathformat(Ns), m.mathformat(ts), m.mathformat(E0res), end-start)
    res = ''
    res += 'Lres[{0}]={1};\n'.format(resi, L)
    res += 'sweeps[{0}]={1};\n'.format(resi, sweeps)
    res += 'maxstates[{0}]={1};\n'.format(resi, maxstates)
    res += 'nmax[{0}]={1};\n'.format(resi, nmax)
    res += 'Nres[{0}]={1};\n'.format(resi, mathformat(Ns))
    res += 'tres[{0}]={1};\n'.format(resi, mathformat(ts))
    res += 'E0res[{0}]={1};\n'.format(resi, mathformat(E0res))
    res += 'nres[{0}]={1};\n'.format(resi, mathformat(nres))
    res += 'n2res[{0}]={1};\n'.format(resi, mathformat(n2res))
    res += 'Cres[{0}]={1};\n'.format(resi, mathformat(Cres))
    res += 'runtime[{0}]={1};\n'.format(resi, end - start)
    resf.write(res)

    gtk.main_quit()

def startmain():
    main_thread = threading.Thread(target=runmain);
    main_thread.start()
    return False

if __name__ == '__main__':
    os.chdir(bhdir)
    [ os.remove(f) for f in os.listdir(".") ]
    gobject.timeout_add(1000, startmain)
    gtk.gdk.threads_init()
    gtk.main()