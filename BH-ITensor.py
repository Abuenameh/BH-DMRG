__author__ = 'Abuenameh'

import sys
import pyalps
import numpy as np
import itertools
import concurrent.futures
import datetime
# import gtk
# import gobject
import threading
import os
import tempfile
import shutil
import subprocess
import time
from mathematica import mathformat
from switch import switch
from speed import gprogress
from multiprocessing import Process, Pipe
from subprocess import PIPE, Popen

try:
    import cPickle as pickle
except:
    import pickle

numthreads = 1

L = 12
nmax = 7

nsweeps = 5

maxm = [20, 20, 100, 100, 200]
#minm = [20, 20, 20, 20, 20]
minm = [20]
cutoff = [1e-10]
niter = [4, 3, 2]
noise = [1e-7, 1e-8, 0]

maxm += [maxm[-1]] * (nsweeps - len(maxm))
minm += [minm[-1]] * (nsweeps - len(minm))
cutoff += [cutoff[-1]] * (nsweeps - len(cutoff))
niter += [niter[-1]] * (nsweeps - len(niter))
noise += [noise[-1]] * (nsweeps - len(noise))

maxm = [str(i) for i in maxm]
minm = [str(i) for i in minm]
cutoff = [str(i) for i in cutoff]
niter = [str(i) for i in niter]
noise = [str(i) for i in noise]

quiet = 'no'

seed = 100
neigen = 1

sweepsTable = 'maxm minm cutoff niter noise\n'
for row in zip(maxm, minm, cutoff, niter, noise):
    sweepsTable += ' '.join(row) + '\n'

# print(sweepsTable)

parametersString = '''
parameters {{{{
    L = {0}
    nmax = {1}
    nsweeps = {2}

    sweeps {{{{
        {3}
    }}}}

    quiet = {4}

    t = {{0}}
    N = {{1}}
}}}}
'''.format(L, nmax, nsweeps, sweepsTable, quiet)

#def createInputFile():


# def writeInputFile(t, N, it, iN):
#     inputFile = open('itensor.{0}.{1}.in'.format(it, iN), 'w')
#     inputFile.write(parametersString.format(t, N))

# def poll(pipe, prog):
#     if pipe.poll():
#         pipe.recv()
#         prog.next()
#     return True
#
# def progressbar(pipe):
#     len = pipe.recv()
#     # prog = gprogress(range(len), size=len).__iter__()
#     # gobject.timeout_add(1, poll, pipe, prog)
#     gtk.main()

appdir = '/Users/Abuenameh/Projects/ITensorDMRG/Release/'

if sys.platform == 'darwin':
    bhdir = '/tmp/BH-DMRG'
elif sys.platform == 'linux2':
    bhdir = '/mnt/BH-DMRG'
elif sys.platform == 'win32':
    bhdir = tempfile.mkdtemp()

def rundmrg(it, t, iN, N):
    inputFile = open('itensor.{0}.{1}.in'.format(it, iN), 'w')
    inputFile.write(parametersString.format(t, N))
    inputFile.close()
    outputFile = open('itensor.{0}.{1}.out'.format(it, iN), 'w')
    print(subprocess.list2cmdline([appdir + 'ITensorDMRG', inputFile.name]))
    subprocess.call(subprocess.list2cmdline([appdir + 'ITensorDMRG', inputFile.name]), shell=True)

def runmain(pipe):
    ts = np.linspace(0.01, 0.3, 15).tolist()
    # ti = int(sys.argv[5])
    # if ti >= 0:
    #     ts = [ts[ti]]
    Ns = range(1, 2 * L + 1, 1)
    ts = np.linspace(0.01, 0.3, 1).tolist()
    Ns = [12]
    # Ns = range(1, 5, 1)

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

    """mindims = [len(ts), len(Ns)]
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
    # E0res = [[[] for j in range(len(Ns))] for k in range(len(ts))]"""

    start = datetime.datetime.now()

    with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
        # futures = [executor.submit(rundmrg, i, tN[0][0], tN[0][1], tN[1][0], tN[1][1]) for i, tN in
        #            enumerate(zip(itertools.product(ts, Ns), itertools.product(range(0, len(ts)), range(0, len(Ns)))))]
        futures = [executor.submit(rundmrg, it, t, iN, N) for (it, t), (iN, N) in
                   itertools.product(enumerate(ts), enumerate(Ns))]
        # pipe.send(len(futures))
        pickle.dump(len(futures), pipe)
        for future in concurrent.futures.as_completed(futures):
            pickle.dump(1, pipe)
            # pipe.send(1)

    """ip = np.zeros([len(ts), len(Ns)])

    data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=filenameprefix))
    for d in data:
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
                if case('Correlation function'):
                    # for x, y in zip(s.x, s.y[0]):
                    #     Cres[it][iN][ip][tuple(x)] = y
                    for ieig, sy in enumerate(s.y):
                        for x, y in zip(s.x, sy):
                            Cres[it][iN][ieig][tuple(x)] = y
                    break
        for ieig in range(neigen):
            Cres[it][iN][ieig][range(L), range(L)] = nres[it][iN][ieig]
            cres[it][iN][ieig] = Cres[it][iN][ieig] / np.sqrt(np.outer(nres[it][iN][ieig], nres[it][iN][ieig]))

    try:
        for it in range(len(ts)):
            for iN in range(len(Ns)):
                m = min(E0res[it][iN])
                ieig = np.where(E0res[it][iN] == m)[0][0]
                truncmin[it][iN] = trunc[it][iN][ieig]
                E0minres[it][iN] = E0res[it][iN][ieig]
                nminres[it][iN] = nres[it][iN][ieig]
                n2minres[it][iN] = n2res[it][iN][ieig]
                Cminres[it][iN] = Cres[it][iN][ieig]
                cminres[it][iN] = cres[it][iN][ieig]
    except Exception as e:
        print(e.message)"""


    end = datetime.datetime.now()

    """resi = sys.argv[1]
    if sys.platform == 'darwin':
        resfile = '/Users/Abuenameh/Documents/Simulation Results/BH-DMRG/res.' + str(resi) + '.txt'
    elif sys.platform == 'linux2':
        resfile = '/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/BH-DMRG/res.' + str(resi) + '.txt'
    elif sys.platform == 'win32':
        resfile = 'C:/Users/abuenameh/Dropbox/Server/BH-DMRG/res.' + str(resi) + '.txt'
    resf = open(resfile, 'w')
    res = ''
    res += 'neigen[{0}]={1};\n'.format(resi, neigen)
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
    res += 'E0minres[{0}]={1};\n'.format(resi, mathformat(E0minres))
    res += 'nminres[{0}]={1};\n'.format(resi, mathformat(nminres))
    res += 'n2minres[{0}]={1};\n'.format(resi, mathformat(n2minres))
    res += 'Cminres[{0}]={1};\n'.format(resi, mathformat(Cminres))
    res += 'cminres[{0}]={1};\n'.format(resi, mathformat(cminres))
    res += 'runtime[{0}]=\"{1}\";\n'.format(resi, end - start)
    resf.write(res)"""

    # gtk.main_quit()


def startmain(pipe):
    main_thread = threading.Thread(target=runmain, args=(pipe,));
    main_thread.start()
    return False


if __name__ == '__main__':
    os.chdir(bhdir)
    [os.remove(f) for f in os.listdir(".")]
    proc = Popen(['python', os.path.dirname(os.path.realpath(__file__)) + '/ProgressDialog.py'], stdin=PIPE)
    pipe = proc.stdin
    runmain(pipe)
    # pipein, pipeout = Pipe()
    # proc = Process(target=progressbar, args=(pipein,))
    # proc.start()
    # gtk.gdk.threads_init()
    # gobject.timeout_add(1000, startmain, pipe)
    # gtk.main()
    proc.terminate()
    if sys.platform == 'win32':
        os.chdir(os.path.expanduser('~'))
        shutil.rmtree(bhdir)
