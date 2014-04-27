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
from progressbar import *
from speed import *

numthreads = 15

L = 50
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

def rundmrg(i, t, N):
    filenameprefix = 'files/BH_' + str(i)
    parms = [dict(parmsbase.items() + { 'N_total' : N, 't' : t }.items())]
    input_file = pyalps.writeInputFiles(filenameprefix, parms)
    res = pyalps.runApplication('/opt/alps/bin/dmrg', input_file, writexml=True)
    data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=filenameprefix))
    for s in data[0]:
        if s.props['observable'] == 'Energy':
            E0 = s.y[0]
    return [t, N, E0]


def runmain():
    Ns = range(1, 2*L, 1)
    ts = np.linspace(0.01, 0.3, 100).tolist()

    E0res = np.zeros([len(ts), len(Ns)])
    E0res.fill(np.NaN)

    start = datetime.datetime.now()

    with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
        futures = [executor.submit(rundmrg, i, tN[0], tN[1]) for i, tN in enumerate(itertools.product(ts, Ns))]
        for future in gprogress(concurrent.futures.as_completed(futures), size=len(futures)):
            try:
                res = future.result()
            except Exception as exc:
                print exc
            else:
                E0res[ts.index(res[0])][Ns.index(res[1])] = res[2]

    end = datetime.datetime.now()
    print end - start

    resi = sys.argv[1]
    resfile = '/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/BH-DMRG/res.' + str(resi) + '.txt'
    # resfile = '/Users/Abuenameh/Documents/Simulation Results/BH-DMRG/res.' + str(resi) + '.txt'
    resf = open(resfile, 'w')
    res = 'Lres[{0}]={1};\nsweeps[{0}]={2};\nmaxstates[{0}]={3};\nNmax[{0}]={4};\nNres[{0}]={5};\ntres[{0}]={6};\nE0res[{0}]={7};\nruntime[{0}]=\"{8}\";\n'.format(resi, L, sweeps, maxstates, Nmax, m.mathformat(Ns), m.mathformat(ts), m.mathformat(E0res), end-start)
    resf.write(res)

    gtk.main_quit()

def startmain():
    main_thread = threading.Thread(target=runmain);
    main_thread.start()
    return False

if __name__ == '__main__':
    gobject.timeout_add(1000, startmain)
    gtk.gdk.threads_init()
    gtk.main()