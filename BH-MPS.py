from __future__ import print_function
from __future__ import division

__author__ = 'abuenameh'

import sys
import pyalps
import numpy as np
import itertools
import concurrent.futures
import datetime
import threading
import os
import tempfile
import shutil
from mathematica import mathformat
from switch import switch
from subprocess import PIPE, Popen

try:
    import cPickle as pickle
except:
    import pickle

if sys.platform == 'darwin':
    numthreads = 6
elif sys.platform == 'linux2':
    numthreads = 15
else:
    numthreads = 1

appname = 'dmrg'
appname = 'mps_optim'

L = 20
sweeps = 20
maxstates = 200  # 1000
warmup = 100
nmax = 7
truncerror = 0  # 1e-10
seed = 100

if len(sys.argv) < 5:
    print('Insufficient number of command line arguments.')
    quit(1)

delta = float(sys.argv[2])
if delta == 0:
    lattice = "open chain lattice"
else:
    lattice = "inhomogeneous chain lattice"

if sys.platform == 'darwin':
    bhdir = '/tmp/BH-MPS'
elif sys.platform == 'linux2':
    bhdir = '/tmp/BH-MPS'
elif sys.platform == 'win32':
    bhdir = tempfile.mkdtemp()

filenameprefix = ''

parmsbase = {
    # 'seed': seed,
    'TEMP_DIRECTORY': bhdir,
    'storagedir': bhdir,
    'MEASURE_LOCAL[Local density]': "n",
    'MEASURE_LOCAL[Local density squared]': "n2",
    'MEASURE_CORRELATIONS[Onebody density matrix]': "bdag:b",
    'LATTICE': lattice,
    'MODEL': "boson Hubbard",
    'CONSERVED_QUANTUMNUMBERS': 'N',
    'SWEEPS': sweeps,
    'L': L,
    'MAXSTATES': maxstates,
    # 'NUM_WARMUP_STATES': warmup,
    'Nmax': nmax,
    'U': 1
}

if truncerror > 0:
    parmsbase['TRUNCATION_ERROR'] = truncerror

if delta > 0:
    mu = 0
    # np.random.seed(int(sys.argv[3]))
    # mu = delta * 2 * np.random.random(L) - delta
    # parmsbase['mu'] = 'sin(x)'

    # parmsbase['mu'] = 'get(x,' + ",".join([str(mui) for mui in mu]) + ')'
    # parmsbase['x'] = '1'
    # parmsbase['mu'] = 'get(x,1,2,3,4)'

    # parmsbase['mu'] = 'get(x,' + ",".join(str(mui)) + ')'
    # parmsbase['delta'] = delta
    # parmsbase['mu'] = 'get(x,0.0493155, -0.0900821, -0.303556, 0.129114, 0.272998, -0.211608, \
    # 0.112826, 0.0688004, -0.215461, 0.307766)'
    # parmsbase['mu'] = 'delta*2*(random() - 0.5)'
else:
    mu = 0

seed = int(sys.argv[3])

resfile = ''
resf = ''
resipath = ''


def poll(pipe, prog):
    if pipe.poll():
        pipe.recv()
        prog.next()
    return True


def app(app):
    if sys.platform == 'win32':
        return 'C:/ALPS/bin/' + app
    else:
        return '/opt/alps/bin/' + app


speckles = {}


def speckle(W):
    if speckles.has_key(W):
        return speckles[W]

    np.random.seed(seed)

    FFTD = 200
    FFTL = int(delta * FFTD)

    A = (4 / np.pi) * (W / FFTD)
    a = [[A * np.exp(2 * np.pi * np.random.random() * 1j) if (i * i + j * j < 0.25 * FFTD * FFTD) else 0 for i in
          range(-FFTL // 2, FFTL // 2, 1)] for j in range(-FFTL // 2, FFTL // 2, 1)]

    b = np.fft.fft2(a)
    s = np.real(b * np.conj(b))
    s2 = np.sqrt(s)
    speckleW = s2.flatten()[0:L]
    speckles[W] = speckleW
    return speckleW


N = 1000;
g13 = 2.5e9;
g24 = 2.5e9;
Delta = -2.0e10;
alpha = 1.1e7;

Ng = np.sqrt(N) * g13;

def JW(W):
    lenW = len(W)
    J = np.zeros(lenW)
    for i in range(0, lenW-1):
        J[i] = alpha * W[i] * W[i+1] / (np.sqrt(Ng * Ng + W[i] * W[i]) * np.sqrt(Ng * Ng + W[i+1] * W[i+1]))
    J[lenW-1] = alpha * W[lenW-1] * W[0] / (np.sqrt(Ng * Ng + W[lenW-1] * W[lenW-1]) * np.sqrt(Ng * Ng + W[0] * W[0]))
    return J


def UW(W):
    return -2*(g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W))


def rundmrg(i, t, N, it, iN):
    # parms = [dict(parmsbase.items() + {'N_total': N, 't': t, 'it': it, 'iN': iN}.items())]
    parms = [dict(parmsbase.items() + {'N_total': N, 't': 'get(x,' + ",".join([str(Ji) for Ji in JW(speckle(t))]) + ')',
                                       'U': 'get(x,' + ",".join([str(Ui) for Ui in UW(speckle(t))]) + ')', 'it': it,
                                       'iN': iN}.items())]
    # parms = [x for x in itertools.chain(parms, parms)]
    # parms = [dict(parm.items() + {'ip': j, 'seed': seed0 + j}.items()) for j, parm in enumerate(parms)]
    input_file = pyalps.writeInputFiles(filenameprefix + str(i), parms)
    pyalps.runApplication(app(appname), input_file, writexml=True)


def runmain(pipe):
    # ts = np.linspace(0.05, 0.3, 15).tolist()
    ts = np.linspace(4e10, 2.5e11, 3).tolist()
    ti = int(sys.argv[4])
    if ti >= 0:
        ts = [ts[ti]]
    ts = [11e10]
    ts = [2.5e11]
    # ts = [np.linspace(0.01, 0.3, 10).tolist()[2]]
    # ts = [0.3]
    # ts = np.linspace(0.3, 0.3, 1).tolist()
    Ns = range(1, 2 * L + 1, 1)
    # Ns = [1]
    # Ns = range(1,15,1)
    # Ns = range(1,16,1)
    # Ns = range(1, L, 1)
    # Ns = range(L+1, 2*L+1, 1)
    # Ns = [ 16 ]
    # Ns = range(3,17,1)
    # Ns = range(1, 16, 1)
    # Ns = range(1,7,1)
    # Ns = [7]
    # Ns = [1,2,3,4,5,6]
    # Ns = [1]
    # Ns = range(7,13,1)
    # Ns = range(1,13,1)
    # Ns = [6,7,8,9]
    # Ns = range(1, L+1, 1)
    # Ns = range(L+1,2*L+1,1)
    # Ns = [L+1,L+2]
    # Ns = [L+1]
    # Do L+2 at some point

    dims = [len(ts), len(Ns)]
    ndims = dims + [L]
    Cdims = dims + [L, L]

    trunc = np.zeros(dims)
    E0res = np.zeros(dims)
    nres = np.zeros(ndims)
    n2res = np.zeros(ndims)
    Cres = np.zeros(Cdims)
    cres = np.zeros(Cdims)

    trunc.fill(np.NaN)
    E0res.fill(np.NaN)
    nres.fill(np.NaN)
    n2res.fill(np.NaN)
    Cres.fill(np.NaN)
    cres.fill(np.NaN)

    mindims = [len(ts), len(Ns)]
    nmindims = mindims + [L]
    Cmindims = mindims + [L, L]

    truncmin = np.zeros(mindims)
    E0minres = np.zeros(mindims)
    nminres = np.zeros(nmindims)
    n2minres = np.zeros(nmindims)
    Cminres = np.zeros(Cmindims)
    cminres = np.zeros(Cmindims)

    truncmin.fill(np.NaN)
    E0minres.fill(np.NaN)
    nminres.fill(np.NaN)
    n2minres.fill(np.NaN)
    Cminres.fill(np.NaN)
    cminres.fill(np.NaN)


    # E0res = [[[np.NaN for i in range(reps)] for j in range(len(Ns))] for k in range(len(ts))]
    # E0res = [[[] for j in range(len(Ns))] for k in range(len(ts))]

    start = datetime.datetime.now()

    with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
        futures = [executor.submit(rundmrg, i, tN[0][0], tN[0][1], tN[1][0], tN[1][1]) for i, tN in
                   enumerate(zip(itertools.product(ts, Ns), itertools.product(range(0, len(ts)), range(0, len(Ns)))))]
        pickle.dump(len(futures), pipe)
        for future in concurrent.futures.as_completed(futures):
            future.result()
            pickle.dump(1, pipe)

    ip = np.zeros([len(ts), len(Ns)])

    data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=filenameprefix))
    for d in data:
        # try:
        it = int(d[0].props['it'])
        iN = int(d[0].props['iN'])
        # ip = int(d[0].props['ip'])
        for s in d:
            for case in switch(s.props['observable']):
                if case('Truncation error'):
                    # trunc[it][iN][ip] = s.y[0]
                    trunc[it][iN] = s.y
                    break
                if case('Energy'):
                    # E0res[it][iN][ip] = s.y[0]
                    E0res[it][iN] = s.y
                    break
                if case('Local density'):
                    # nres[it][iN][ip] = s.y[0]
                    nres[it][iN] = s.y
                    break
                if case('Local density squared'):
                    # n2res[it][iN][ip] = s.y[0]
                    n2res[it][iN] = s.y
                    break
                if case('Onebody density matrix'):
                    # for x, y in zip(s.x, s.y[0]):
                    # Cres[it][iN][ip][tuple(x)] = y
                    for x, y in zip(s.x, s.y[0]):
                        Cres[it][iN][tuple(x)] = y
                    # for ieig, sy in enumerate(s.y):
                    #     for x, y in zip(s.x, sy):
                    #         Cres[it][iN][ieig][tuple(x)] = y
                    break
        Cres[it][iN][range(L), range(L)] = nres[it][iN]
        cres[it][iN] = Cres[it][iN] / np.sqrt(np.outer(nres[it][iN], nres[it][iN]))
        # for ieig in range(neigen):
        #     Cres[it][iN][ieig][range(L), range(L)] = nres[it][iN][ieig]
        #     cres[it][iN][ieig] = Cres[it][iN][ieig] / np.sqrt(np.outer(nres[it][iN][ieig], nres[it][iN][ieig]))
        # except Exception as e:
        #     print(e.message)

    # for it in range(len(ts)):
    #     for iN in range(len(Ns)):
    #         try:
    #             m = min(E0res[it][iN])
    #             ieig = np.where(E0res[it][iN] == m)[0][0]
    #             truncmin[it][iN] = trunc[it][iN][ieig]
    #             E0minres[it][iN] = E0res[it][iN][ieig]
    #             nminres[it][iN] = nres[it][iN][ieig]
    #             n2minres[it][iN] = n2res[it][iN][ieig]
    #             Cminres[it][iN] = Cres[it][iN][ieig]
    #             cminres[it][iN] = cres[it][iN][ieig]
    #         except Exception as e:
    #             print(e.message)

    end = datetime.datetime.now()

    res = ''
    res += 'Wres[{0}]={1};\n'.format(resi, mathformat([speckle(Wi) for Wi in ts]))
    res += 'Jres[{0}]={1};\n'.format(resi, mathformat([JW(speckle(Wi)) for Wi in ts]))
    res += 'Ures[{0}]={1};\n'.format(resi, mathformat([UW(speckle(Wi)) for Wi in ts]))
    res += 'Wmres[{0}]={1};\n'.format(resi, mathformat([Wi for Wi in ts]))
    res += 'Jmres[{0}]={1};\n'.format(resi, mathformat([JW(np.array([Wi,Wi]))[0] for Wi in ts]))
    res += 'Umres[{0}]={1};\n'.format(resi, mathformat([UW(np.array([Wi]))[0] for Wi in ts]))
    # res += 'neigen[{0}]={1};\n'.format(resi, neigen)
    res += 'delta[{0}]={1};\n'.format(resi, delta)
    res += 'trunc[{0}]={1};\n'.format(resi, mathformat(trunc))
    res += 'Lres[{0}]={1};\n'.format(resi, L)
    res += 'sweeps[{0}]={1};\n'.format(resi, sweeps)
    res += 'maxstates[{0}]={1};\n'.format(resi, maxstates)
    res += 'warmup[{0}]={1};\n'.format(resi, warmup)
    res += 'truncerror[{0}]={1};\n'.format(resi, truncerror)
    res += 'nmax[{0}]={1};\n'.format(resi, nmax)
    res += 'Nres[{0}]={1};\n'.format(resi, mathformat(Ns))
    res += 'tres[{0}]={1};\n'.format(resi, mathformat(ts))
    res += 'mures[{0}]={1};\n'.format(resi, mathformat(mu))
    res += 'E0res[{0}]={1};\n'.format(resi, mathformat(E0res))
    res += 'nres[{0}]={1};\n'.format(resi, mathformat(nres))
    res += 'n2res[{0}]={1};\n'.format(resi, mathformat(n2res))
    res += 'Cres[{0}]={1};\n'.format(resi, mathformat(Cres))
    res += 'cres[{0}]={1};\n'.format(resi, mathformat(cres))
    res += 'truncmin[{0}]={1};\n'.format(resi, mathformat(truncmin))
    # res += 'E0minres[{0}]={1};\n'.format(resi, mathformat(E0minres))
    # res += 'nminres[{0}]={1};\n'.format(resi, mathformat(nminres))
    # res += 'n2minres[{0}]={1};\n'.format(resi, mathformat(n2minres))
    # res += 'Cminres[{0}]={1};\n'.format(resi, mathformat(Cminres))
    # res += 'cminres[{0}]={1};\n'.format(resi, mathformat(cminres))
    res += 'runtime[{0}]=\"{1}\";\n'.format(resi, end - start)

    resf.write(res)

    shutil.make_archive(resipath, 'zip', resipath)
    shutil.rmtree(resipath)


if __name__ == '__main__':
    resi = int(sys.argv[1])
    if sys.platform == 'darwin':
        respath = '/Users/Abuenameh/Documents/Simulation Results/BH-MPS/'
    elif sys.platform == 'linux2':
        respath = '/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/BH-MPS/'
    elif sys.platform == 'win32':
        respath = 'C:/Users/abuenameh/Dropbox/Server/BH-DMRG/'
    if not os.path.exists(respath):
        try:
            os.makedirs(respath)
        except:
            pass
    resipath = respath + 'res.' + str(resi)
    resfile = resipath + '.txt'
    while os.path.isfile(resfile):
        resi += 1
        resipath = respath + 'res.' + str(resi)
        resfile = resipath + '.txt'
    resf = open(resfile, 'w')
    datadir = resipath + '/'
    os.makedirs(datadir)
    filenameprefix = datadir + 'BH_'

    proc = Popen(['python', os.path.dirname(os.path.realpath(__file__)) + '/ProgressDialog.py'], stdin=PIPE)
    pipe = proc.stdin
    runmain(pipe)
    proc.terminate()
