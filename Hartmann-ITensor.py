__author__ = 'Abuenameh'

import sys
import numpy as np
import itertools
import concurrent.futures
import datetime
import threading
import os
import tempfile
import shutil
import subprocess
import time
from mathematica import mathformat, timedeltaformat
from switch import switch
from subprocess import PIPE, Popen

try:
    import cPickle as pickle
except:
    import pickle

numthreads = 6

L = 20
nmax = 7

U = 1

N = 1000
g13 = 2.5e9
g24 = 2.5e9
delta = 1e12
Delta = 

nsweeps = 5
errgoal = -1#1e-16

maxm = [20, 20, 100, 100, 200]
# maxm = [300]
minm = [20]
cutoff = [1e-10]
niter = [4, 3, 2]
# niter = [4]
noise = [1e-5, 1e-6, 1e-7, 1e-8, 0]
# noise = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-8,1e-8,0]
# noise = [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,0]

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

quiet = 'yes'

seed = 100
neigen = 1

if len(sys.argv) < 4:
    print('Insufficient number of command line arguments.')
    quit(1)

delta = float(sys.argv[2])

if delta > 0:
    np.random.seed(int(sys.argv[3]))
    mu = delta*2*np.random.random(L) - delta
    # mu = [0.0244067519637,0.107594683186,0.0513816880358,0.0224415914984,-0.0381726003305,0.0729470565333,-0.0312063943687,0.195886500391,0.231831380251,-0.0582792405871,0.145862519041,0.0144474598765]
    mustr = ",".join([str(mui) for mui in mu])
else:
    mu = 0
    mustr = str(mu)

sweepsTable = 'maxm minm cutoff niter noise\n'
for row in zip(maxm, minm, cutoff, niter, noise):
    sweepsTable += ' '.join(row) + '\n'

parametersString = '''
parameters {{{{
    L = {0}
    nmax = {1}
    nsweeps = {2}
    errgoal = {3}

    sweeps {{{{
        {4}
    }}}}

    quiet = {5}

    t = {{0}}
    N = {{1}}
    U = {{2}}
    mu = {{3}}
}}}}
'''.format(L, nmax, nsweeps, errgoal, sweepsTable, quiet)

if sys.platform == 'darwin':
    appdir = '/Users/Abuenameh/Projects/ITensorDMRG/Release/'
elif sys.platform == 'win32':
    appdir = 'C:/Users/abuenameh/Documents/Projects/ITensorDMRG/Release/'

if sys.platform == 'darwin':
    bhdir = '/tmp/BH-DMRG'
elif sys.platform == 'linux2':
    bhdir = '/mnt/BH-DMRG'
elif sys.platform == 'win32':
    bhdir = tempfile.mkdtemp()

if not os.path.isdir(bhdir):
    os.makedirs(bhdir)

def rundmrg(it, t, iN, N):
    inputFile = open('itensor.{0}.{1}.in'.format(it, iN), 'w')
    inputFile.write(parametersString.format(t, N, U, mustr))
    inputFile.close()
    outputFileName = 'itensor.{0}.{1}.out'.format(it, iN)
    # print(subprocess.list2cmdline([appdir + 'ITensorDMRG', inputFile.name]))
    subprocess.call(subprocess.list2cmdline([appdir + 'ITensorDMRG', inputFile.name, outputFileName]), shell=True)

def pad(a, size, v):
    l = len(a)
    return np.concatenate((a,[v]*(size-l)))

def run(pipe):
    ts = np.linspace(0.01, 0.3, 15).tolist()
    # ti = int(sys.argv[5])
    # if ti >= 0:
    #     ts = [ts[ti]]
    Ns = range(1, 2 * L + 1, 1)
    # ts = np.linspace(0.01, 0.3, 1).tolist()
    # Ns = [4]
    # Ns = [8]
    # Ns = range(1, 5, 1)
    # Ns = range(1, 13, 1)

    dims = [len(ts), len(Ns)]
    Edims = dims + [nsweeps]
    ndims = dims + [L]
    Cdims = dims + [L, L]

    trunc = np.zeros(dims)
    E0res = np.zeros(dims)
    runtimeres = np.zeros(dims)
    Eires = np.zeros(Edims)
    nres = np.zeros(ndims)
    n2res = np.zeros(ndims)
    Cres = np.zeros(Cdims)
    cres = np.zeros(Cdims)

    trunc.fill(np.NaN)
    E0res.fill(np.NaN)
    runtimeres.fill(np.NaN)
    Eires.fill(np.NaN)
    nres.fill(np.NaN)
    n2res.fill(np.NaN)
    Cres.fill(np.NaN)
    cres.fill(np.NaN)

    start = datetime.datetime.now()

    with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
        futures = [executor.submit(rundmrg, it, t, iN, N) for (it, t), (iN, N) in
                   itertools.product(enumerate(ts), enumerate(Ns))]
        pickle.dump(len(futures), pipe)
        for future in concurrent.futures.as_completed(futures):
            pickle.dump(1, pipe)

    for it, iN in itertools.product(range(len(ts)), range(len(Ns))):
        try:
            outputFile = open('itensor.{0}.{1}.out'.format(it, iN), 'r')
            for line in outputFile:
                lineSplit = line.split()
                obs = lineSplit[0]
                val = np.array([float(s) for s in lineSplit[1:]])
                for case in switch(obs):
                    if case('Ei'):
                        Eires[it][iN] = pad(val, nsweeps, np.NaN)
                        break
                    if case('E0'):
                        E0res[it][iN] = val[0]
                        break
                    if case('n'):
                        nres[it][iN] = val
                        break
                    if case('n2'):
                        n2res[it][iN] = val
                        break
                    if case('C'):
                        Cres[it][iN] = np.split(val,L)
                        break
                    if case('runtime'):
                        runtimeres[it][iN] = val[0]
                        break
            outputFile.close()
        except Exception as e:
            print(e.message)


    end = datetime.datetime.now()

    resi = sys.argv[1]
    if sys.platform == 'darwin':
        resfile = '/Users/Abuenameh/Documents/Simulation Results/BH-ITensor-DMRG/res.' + str(resi) + '.txt'
    elif sys.platform == 'linux2':
        resfile = '/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/BH-ITensor-DMRG/res.' + str(resi) + '.txt'
    elif sys.platform == 'win32':
        resfile = 'C:/Users/abuenameh/Dropbox/Server/BH-ITensor-DMRG/res.' + str(resi) + '.txt'
    if not os.path.isdir(os.path.dirname(resfile)):
        os.makedirs(os.path.dirname(resfile))
    resf = open(resfile, 'w')
    res = ''
    res += 'delta[{0}]={1};\n'.format(resi, delta)
    res += 'Lres[{0}]={1};\n'.format(resi, L)
    res += 'nsweeps[{0}]={1};\n'.format(resi, nsweeps)
    res += 'errgoal[{0}]={1};\n'.format(resi, mathformat(errgoal))
    res += 'maxm[{0}]={1};\n'.format(resi, mathformat(maxm))
    res += 'minm[{0}]={1};\n'.format(resi, mathformat(minm))
    res += 'cutoff[{0}]={1};\n'.format(resi, mathformat(cutoff))
    res += 'niter[{0}]={1};\n'.format(resi, mathformat(niter))
    res += 'noise[{0}]={1};\n'.format(resi, mathformat(noise))
    res += 'nmax[{0}]={1};\n'.format(resi, nmax)
    res += 'Nres[{0}]={1};\n'.format(resi, mathformat(Ns))
    res += 'tres[{0}]={1};\n'.format(resi, mathformat(ts))
    res += 'mures[{0}]={1};\n'.format(resi, mathformat(mu))
    res += 'Eires[{0}]={1};\n'.format(resi, mathformat(Eires))
    res += 'E0res[{0}]={1};\n'.format(resi, mathformat(E0res))
    res += 'nres[{0}]={1};\n'.format(resi, mathformat(nres))
    res += 'n2res[{0}]={1};\n'.format(resi, mathformat(n2res))
    res += 'Cres[{0}]={1};\n'.format(resi, mathformat(Cres))
    try:
        res += 'runtimei[{0}]={1};\n'.format(resi, timedeltaformat(runtimeres))
    except:
        pass
    res += 'runtime[{0}]=\"{1}\";\n'.format(resi, end - start)
    resf.write(res)


if __name__ == '__main__':
    os.chdir(bhdir)
    [os.remove(f) for f in os.listdir(".")]
    proc = Popen(['python', os.path.dirname(os.path.realpath(__file__)) + '/ProgressDialog.py'], stdin=PIPE)
    pipe = proc.stdin
    run(pipe)
    proc.terminate()
    if sys.platform == 'win32':
        os.chdir(os.path.expanduser('~'))
        shutil.rmtree(bhdir)
