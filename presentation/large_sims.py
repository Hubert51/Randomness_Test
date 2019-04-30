import math
import os
import re

import utils.PyRandomUtils as pru
import numpy as np
import utils.CardUtils as cu

class countwrapper(pru.PRNG):
    def __init__(self, gen):
        self.gen = gen
        self.randcount = 0

    def seed(self, x):
        self.gen.seed(x)
        self.randcount = 0

    def reset(self):
        self.randcount = 0
    def rand(self):
        self.randcount += 1
        return self.gen.rand()


def run_test(gen, n=10**6):
    ts = pru.make_ts_no_batch(gen, n)
    means, sigma = find_means(ts)
    ret = []
    for i in range(len(means)):
        ret.append(((means[i] - cu.theoretical_probabilities[i]) / (sigma[i] / math.sqrt(n)))
                   if sigma[i] else float('inf') * np.sign(means[i]-cu.theoretical_probabilities[i]))
    return ret

def find_means(ts):
    means = np.apply_along_axis(np.mean, 1, ts)
    sigma = np.apply_along_axis(np.std, 1, ts)
    return means, sigma

def aggregate_statistics(counts, means, devs, ddof=0):

    r = range(len(counts))
    n = sum(counts)
    mean = 1 / n * sum([counts[i] * means[i] for i in r])

    s = 0
    for i in r:
        s += counts[i] * (means[i] - mean) ** 2 + (counts[i])*(devs[i]**2)
    dev = np.sqrt(1 / (n-ddof) * s)

    return mean, dev

def run_test_with_saving(gen, n, saves, filename, verbose=True, i=0):
    run = n//saves
    ret = []
    counts = []
    means=[]
    devs=[]
    final=False
    with open(filename, 'w') as fout:
        while not final:
            if i + run < n:
                result = pru.make_ts_no_batch(gen, run)
            else:
                final=True
                result = pru.make_ts_no_batch(gen, n-i)
            mean, std= find_means(result)
            counts.append(len(result))
            if hasattr(gen, "randcount"):
                fout.write("{} [{} rands]: [{}], [{}]\n".format(str(i), gen.randcount, ','.join((str(x) for x in mean)), ','.join((str(x) for x in std))))
            else:
                fout.write("{}: [{}], [{}]\n".format(str(i), ','.join((str(x) for x in mean)), ','.join((str(x) for x in std))))
            means.append(mean)
            devs.append(std)
            i += run
            print("i={}".format(i))

        mean, sigma = aggregate_statistics(counts, means, devs)
        fout.write('\n')
        fout.write("total: {}\n".format(n))
        fout.write("means: [{}]\n".format(','.join((str(x) for x in mean))))
        fout.write("sigma: [{}]\n".format(','.join((str(x) for x in sigma))))
        fout.write("cu   : [{}]\n".format(','.join((str(x) for x in cu.theoretical_probabilities))))
        diff = []
        for i in range(len(cu.theoretical_probabilities)):
            diff.append(((mean[i] - cu.theoretical_probabilities[i]) / (sigma[i] / math.sqrt(n))))



        fout.write("diff : [{}]\n".format(','.join((str(x) for x in diff))))

    return diff


def run_test_file(gen, n, filename):
    ts = pru.make_ts_no_batch(gen, n, prog=1)
    with open(filename, 'w') as fout:
        mean, sigma = find_means(ts)
        fout.write("total: {}\n".format(n))
        fout.write("means: [{}]\n".format(','.join((str(x) for x in mean))))
        fout.write("sigma: [{}]\n".format(','.join((str(x) for x in sigma))))
        fout.write("cu   : [{}]\n".format(','.join((str(x) for x in cu.theoretical_probabilities))))
        diff = []
        for i in range(len(cu.theoretical_probabilities)):
            diff.append(((mean[i] - cu.theoretical_probabilities[i]) / (sigma[i] / math.sqrt(n))))



        fout.write("diff : [{}]\n".format(','.join((str(x) for x in diff))))

    return diff

def fix_file(infile):
    outfile = infile + '_fix'
    p = re.compile(r'\d+ \[\d+ rands\]: (\[.*\]), (\[.*\])')

    n = 10 ** 9

    means = []
    devs = []

    with open(infile) as fin:
        with open(outfile, 'w') as fout:
            for line in fin:
                fout.write(line)
                if line == '\n':
                    continue
                m = p.match(line)
                if not m:
                    #                 print("ERROR: did not match [%s]" % line)
                    break

                means.append(np.array(eval(m.group(1))))
                devs.append(np.array(eval(m.group(2))))

            mean, sigma = aggregate_statistics([10 ** 6] * len(means), means, devs)
            fout.write("means: [{}]\n".format(','.join((str(x) for x in mean))))
            next(fin)
            fout.write("sigma: [{}]\n".format(','.join((str(x) for x in sigma))))
            next(fin)
            fout.write("cu   : [{}]\n".format(','.join((str(x) for x in cu.theoretical_probabilities))))
            diff = []
            for i in range(len(cu.theoretical_probabilities)):
                diff.append(((mean[i] - cu.theoretical_probabilities[i]) / (sigma[i] / math.sqrt(n))))
            fout.write("diff : [{}]\n".format(','.join((str(x) for x in diff))))

def to_latex():
    files = [f for f in os.listdir() if re.match(r'(.*)_1bil.txt$', f)]
    means = []
    sigma = []
    names = []
    for f in files:
        with open(f, 'r') as fin:
            gen = re.match(r'(.*)_1bil.txt$', f).group(1)
            names.append(gen)

            lines = fin.readlines()
            meanline = next((x for x in lines if x.startswith('means:')))
            means.append(eval(meanline[meanline.index(':')+1:]))

            sigmaline = next((x for x in lines if x.startswith('sigma:')))
            sigma.append(eval(sigmaline[sigmaline.index(':')+1:]))

    print("Feature & {\\~ P} & ", end='')
    print(' & '.join(names) + '\\\\')

    for i in range(len(cu.theoretical_probabilities)):
        print("{} & {:6.3} & ".format(cu.feature_string[i], cu.theoretical_probabilities[i]), end='')
        print(" & ".join(("{:6.3}".format(a[i]) for a in means))+' \\\\')

if __name__ == '__main__':
    # gen = pru.PyRandGen(1)
    # res2 = run_test_with_saving(gen, 10 ** 6, 100, 'testMT.txt')
    # gen = pru.PyRandGen(1)
    # res2 = run_test_with_saving(countwrapper(gen), 10 ** 9, 1000, 'MT1bil.txt')
    # gen = pru.LCG_RANDU(1)
    # res2 = run_test_with_saving(countwrapper(gen), 10 ** 9, 1000, 'RANDU1bil.txt')
    # gen = pru.LCG(mod=2**16, a=5, seed=1, c=0)
    # res2 = run_test_with_saving(countwrapper(gen), 10 ** 9, 1000, 'badLCG_1bil.txt')
    # gen = pru.LCG(mod=2**32, a=1664525, c=1013904223, seed=1)
    # res2 = run_test_with_saving(countwrapper(gen), 10 ** 9, 1000, 'goodLCG_1bil.txt')
    # gen = pru.HaltonGen_Deck(batch_size=10**6)
    # res2 = run_test_with_saving(countwrapper(gen), 10 ** 9, 1000, 'halton_1bil.txt')
    # gen = pru.HaltonGen(base=19)
    # res2 = run_test_with_saving(countwrapper(gen), 10 ** 9, 1000, 'vdc19_1bil_app.txt')

    # fix_file('MT1bil.txt')
    # fix_file('RANDU1bil.txt')
    # fix_file('badLCG_1bil.txt')
    # fix_file('goodLCG_1bil.txt')
    # fix_file('halton_1bil.txt')
    # fix_file('vdc19_1bil.txt')


    to_latex()