from __future__ import print_function

__author__ = 'Abuenameh'

import sys
import pyalps
import numpy as np
import itertools
import concurrent.futures
import datetime
import gtk
import gobject
import threading
import os
from mathematica import mathformat
from switch import switch
from speed import gprogress

numthreads = 15

L = 20
sweeps = 10
maxstates = 1000
warmup = 100
nmax = 7
truncerror = 1e-10
perturb = 1000.0#0#1e-10
reps = 1

if len(sys.argv) < 3:
    print('Insufficient number of command line arguments.')
    quit(1)

delta = float(sys.argv[2])
if delta == 0:
    lattice = "open chain lattice"
else:
    lattice = "inhomogeneous chain lattice"

if sys.platform == 'darwin':
    bhdir = '/tmp/BH-DMRG'
elif sys.platform == 'linux2':
    bhdir = '/mnt/BH-DMRG'
filenameprefix = 'BH_'

parmsbase = {
    'TEMP_DIRECTORY': bhdir,
    'MEASURE_LOCAL[Local density]': "n",
    'MEASURE_LOCAL[Local density squared]': "n2",
    'MEASURE_CORRELATIONS[Correlation function]': "bdag:b",
    'LATTICE': lattice,
    'MODEL': "boson Hubbard",
    'CONSERVED_QUANTUMNUMBERS': 'N',
    'SWEEPS': sweeps,
    'NUMBER_EIGENVALUES': 1,
    'L': L,
    'MAXSTATES': maxstates,
    # 'NUM_WARMUP_STATES': warmup,
    'PERTURBATION': perturb,
    'Nmax': nmax,
    'U': 1
}

if truncerror > 0:
    parmsbase['TRUNCATION_ERROR'] = truncerror

if delta > 0:
    np.random.seed(int(sys.argv[3]))
    mu = delta*2*np.random.random(L) - delta
    parmsbase['mu'] = 'get(x,' + ",".join([str(mui) for mui in mu]) + ')'
    # parmsbase['mu'] = 'get(x,' + ",".join(str(mui)) + ')'
    # parmsbase['delta'] = delta
    # parmsbase['mu'] = 'get(x,0.0493155, -0.0900821, -0.303556, 0.129114, 0.272998, -0.211608, \
    # 0.112826, 0.0688004, -0.215461, 0.307766)'
    # parmsbase['mu'] = 'delta*2*(random() - 0.5)'
else:
    mu = 0


def rundmrg(i, t, N, it, iN):
    parms = [dict(parmsbase.items() + {'N_total': N, 't': t, 'it': it, 'iN': iN}.items())]
    # parms = [x for x in itertools.chain(parms, parms)]
    parms = [x for x in itertools.chain.from_iterable(itertools.repeat(parms, reps))]
    parms = [dict(parm.items() + {'ip': j}.items()) for j, parm in enumerate(parms)]
    input_file = pyalps.writeInputFiles(filenameprefix + str(i), parms)
    pyalps.runApplication('/opt/alps/bin/dmrg2', input_file, writexml=True)


def runmain():
    ts = np.linspace(0.01, 0.3, 1).tolist()
    # ts = [np.linspace(0.01, 0.3, 10).tolist()[2]]
    # ts = [0.3]
    # ts = np.linspace(0.3, 0.3, 1).tolist()
    Ns = range(1, 2 * L + 1, 1)
    # Ns = range(1,15,1)
    # Ns = range(1,16,1)
    # Ns = range(1, L, 1)
    # Ns = range(L+1, 2*L+1, 1)
    # Ns = [ 16 ]
    # Ns = range(3,17,1)
    Ns = range(1, 16, 1)

    dims = [len(ts), len(Ns), reps]
    ndims = dims + [L]
    Cdims = dims + [L, L]

    trunc = np.zeros(dims)

    E0res = np.zeros(dims)
    nres = np.zeros(ndims)
    n2res = np.zeros(ndims)
    Cres = np.zeros(Cdims)
    cres = np.zeros(Cdims)

    E0res.fill(np.NaN)
    nres.fill(np.NaN)
    n2res.fill(np.NaN)
    Cres.fill(np.NaN)
    cres.fill(np.NaN)

    # E0res = [[[np.NaN for i in range(reps)] for j in range(len(Ns))] for k in range(len(ts))]
    # E0res = [[[] for j in range(len(Ns))] for k in range(len(ts))]

    start = datetime.datetime.now()

    with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
        futures = [executor.submit(rundmrg, i, tN[0][0], tN[0][1], tN[1][0], tN[1][1]) for i, tN in
                   enumerate(zip(itertools.product(ts, Ns), itertools.product(range(0, len(ts)), range(0, len(Ns)))))]
        for future in gprogress(concurrent.futures.as_completed(futures), size=len(futures)):
            pass

    ip = np.zeros([len(ts), len(Ns)])

    data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=filenameprefix))
    for d in data:
        for s in d:
            it = int(s.props['it'])
            iN = int(s.props['iN'])
            ip = int(s.props['ip'])
            for case in switch(s.props['observable']):
                if case('Truncation error'):
                    trunc[it][iN][ip] = s.y[0]
                    break
                if case('Energy'):
                    E0res[it][iN][ip] = s.y[0]
                    # E0res[it][iN].append(s.y[0])
                    break
                if case('Local density'):
                    nres[it][iN][ip] = s.y[0]
                    break
                if case('Local density squared'):
                    n2res[it][iN][ip] = s.y[0]
                    break
                if case('Correlation function'):
                    Cres[it][iN][ip] = np.split(s.y[0], L)
                    break
            cres[it][iN][ip] = Cres[it][iN][ip] / np.sqrt(np.outer(nres[it][iN][ip], nres[it][iN][ip]))

    end = datetime.datetime.now()

    resi = sys.argv[1]
    if sys.platform == 'darwin':
        resfile = '/Users/Abuenameh/Documents/Simulation Results/BH-DMRG/res.' + str(resi) + '.txt'
    elif sys.platform == 'linux2':
        resfile = '/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/BH-DMRG/res.' + str(resi) + '.txt'
    resf = open(resfile, 'w')
    res = ''
    res += 'delta[{0}]={1};\n'.format(resi, delta)
    res += 'trunc[{0}]={1};\n'.format(resi, mathformat(trunc))
    res += 'Lres[{0}]={1};\n'.format(resi, L)
    res += 'sweeps[{0}]={1};\n'.format(resi, sweeps)
    res += 'maxstates[{0}]={1};\n'.format(resi, maxstates)
    res += 'warmup[{0}]={1};\n'.format(resi, warmup)
    res += 'truncerror[{0}]={1};\n'.format(resi, truncerror)
    res += 'perturb[{0}]={1};\n'.format(resi, perturb)
    res += 'nmax[{0}]={1};\n'.format(resi, nmax)
    res += 'Nres[{0}]={1};\n'.format(resi, mathformat(Ns))
    res += 'tres[{0}]={1};\n'.format(resi, mathformat(ts))
    res += 'mures[{0}]={1};\n'.format(resi, mathformat(mu))
    res += 'E0res[{0}]={1};\n'.format(resi, mathformat(E0res))
    res += 'nres[{0}]={1};\n'.format(resi, mathformat(nres))
    res += 'n2res[{0}]={1};\n'.format(resi, mathformat(n2res))
    res += 'Cres[{0}]={1};\n'.format(resi, mathformat(Cres))
    res += 'cres[{0}]={1};\n'.format(resi, mathformat(cres))
    res += 'runtime[{0}]=\"{1}\";\n'.format(resi, end - start)
    resf.write(res)

    gtk.main_quit()


def startmain():
    main_thread = threading.Thread(target=runmain);
    main_thread.start()
    return False


if __name__ == '__main__':
    os.chdir(bhdir)
    [os.remove(f) for f in os.listdir(".")]
    gobject.timeout_add(1000, startmain)
    gtk.gdk.threads_init()
    gtk.main()