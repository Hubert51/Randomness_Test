import math
import os
import re

import numpy as np
import scipy.stats

import utils.CardUtils as cu
import utils.PyRandomUtils as pru


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

class urand_gen(pru.PRNG):

    def make_buffer(self):
        self.buffer = np.frombuffer(os.urandom(self.bufsize), dtype=np.uint16)

    def __init__(self, bufsize = 1024 * 8):
         self.bufsize = bufsize
         self.make_buffer()
         self.randcount = 0

    def rand(self):
        ret = self.buffer[self.randcount%len(self.buffer)] / np.iinfo(np.uint16).max
        self.randcount += 1
        if self.randcount%len(self.buffer) == 0:
            self.make_buffer()
        return ret

    def seed(self, x):
        pass

    def reset(self):
        pass

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
    alpha = (1-scipy.stats.norm.cdf(5))/2 # 5 sigma
    files = [f for f in os.listdir() if re.match(r'(.*)_1bil.txt$', f)]
    means = []
    sigma = []
    names = []
    significance = []
    for f in files:
        with open(f, 'r') as fin:
            gen = re.match(r'(.*)_1bil.txt$', f).group(1)
            names.append(gen)

            lines = fin.readlines()
            meanline = next((x for x in lines if x.startswith('means:')))
            means.append(eval(meanline[meanline.index(':')+1:]))

            sigmaline = next((x for x in lines if x.startswith('sigma:')))
            sigma.append(eval(sigmaline[sigmaline.index(':')+1:]))

            totalsline = next((x for x in lines if x.startswith('total:')))
            total = (eval(totalsline[totalsline.index(':') + 1:]))

            significant = []

            for i in range(len(cu.theoretical_probabilities)):
                # test for significance on each feature
                tp = cu.theoretical_probabilities[i]
                mean = means[-1][i]
                sigma_ = tp * (1-tp)


                z = abs((mean - tp)) * np.sqrt(total/sigma_)
                p = 1.0 - scipy.stats.norm.cdf(z)
                significant.append(p < alpha)

            significance.append(significant)




    print("Feature & {\\~ P} & ", end='')
    print(' & '.join(names[i] if not any(significance[i]) else '\\bf %s'%names[i] for i in range(len(names))) + '\\\\')

    for i in range(len(cu.theoretical_probabilities)):
        print("{} & {:.4E} & ".format(cu.feature_string[i], cu.theoretical_probabilities[i]), end='')
        to_join = []
        for j in range(len(means)):
            if significance[j][i]:
                    to_join.append('\\textbf{{{:.4E}}}'.format(means[j][i]))
            else:
                to_join.append('{:.4E}' .format(means[j][i]))

        print(" & ".join(to_join)+' \\\\')

def to_latex_deals():
    alpha = (1-scipy.stats.norm.cdf(5))/2 # 5 sigma
    means = [np.array([3.5012218e-01, 4.4482371e-01, 1.6553858e-01, 3.4579728e-02,
       4.5188316e-03, 3.9757960e-04, 1.9394127e-05, 0.0000000e+00,
       0.0000000e+00, 0.0000000e+00, 5.1985960e-02, 3.0596176e-01,
       5.3713006e-01, 1.0492223e-01, 1.3556495e-02, 7.5637095e-04,
       1.2897095e-03, 4.8485319e-05, 2.8495792e-01, 5.1441956e-01,
       1.7615686e-01, 2.4465691e-02], dtype=np.float32),
             ]
    names = ['3_months', '1_year', 'BigDeal']
    significance = []
    for f in files:
        with open(f, 'r') as fin:
            gen = re.match(r'(.*)_1bil.txt$', f).group(1)
            names.append(gen)

            lines = fin.readlines()
            meanline = next((x for x in lines if x.startswith('means:')))
            means.append(eval(meanline[meanline.index(':')+1:]))

            sigmaline = next((x for x in lines if x.startswith('sigma:')))
            sigma.append(eval(sigmaline[sigmaline.index(':')+1:]))

            totalsline = next((x for x in lines if x.startswith('total:')))
            total = (eval(totalsline[totalsline.index(':') + 1:]))

            significant = []

            for i in range(len(cu.theoretical_probabilities)):
                # test for significance on each feature
                tp = cu.theoretical_probabilities[i]
                mean = means[-1][i]
                sigma_ = tp * (1-tp)


                z = abs((mean - tp)) * np.sqrt(total/sigma_)
                p = 1.0 - scipy.stats.norm.cdf(z)
                significant.append(p < alpha)

            significance.append(significant)




    print("Feature & {\\~ P} & ", end='')
    print(' & '.join(names[i] if not any(significance[i]) else '\\bf %s'%names[i] for i in range(len(names))) + '\\\\')

    for i in range(len(cu.theoretical_probabilities)):
        print("{} & {:.4E} & ".format(cu.feature_string[i], cu.theoretical_probabilities[i]), end='')
        to_join = []
        for j in range(len(means)):
            if significance[j][i]:
                    to_join.append('\\textbf{{{:.4E}}}'.format(means[j][i]))
            else:
                to_join.append('{:.4E}' .format(means[j][i]))

        print(" & ".join(to_join)+' \\\\')




if __name__ == '__main__':
    # gen = pru.PyRandGen(1)
    # res2 = run_test_with_saving(gen, 10 ** 6, 100, 'testMT.txt')
    # gen = pru.PyRandGen(1)
    # res2 = run_test_with_saving(countwrapper(gen), 10 ** 9, 1000, 'MT1bil.txt')
    # gen = pru.LCG_RANDU(1)
    # res2 = run_test_with_saving(countwrapper(gen), 10 ** 9, 1000, 'RANDU_1bil.txt')
    # gen = pru.LCG(mod=2**16, a=5, seed=1, c=0)
    # res2 = run_test_with_saving(countwrapper(gen), 10 ** 9, 1000, 'badLCG_1bil.txt')
    # gen = pru.LCG(mod=2**32, a=1664525, c=1013904223, seed=1)
    # res2 = run_test_with_saving(countwrapper(gen), 10 ** 9, 1000, 'goodLCG_1bil.txt')
    # gen = pru.HaltonGen_Deck(batch_size=10**6)
    # res2 = run_test_with_saving(countwrapper(gen), 10 ** 9, 1000, 'halton_1bil.txt')
    # gen = pru.HaltonGen(base=19)
    # res2 = run_test_with_saving(countwrapper(gen), 10 ** 9, 1000, 'vdc19_1bil_app.txt')
    # gen = urand_gen()
    # res2 = run_test_with_saving(gen, 10 ** 8, 1000, 'windows_urand_1bil.txt')

    # fix_file('MT1bil.txt')
    # fix_file('RANDU_1bil.txt')
    # fix_file('badLCG_1bil.txt')
    # fix_file('goodLCG_1bil.txt')
    # fix_file('halton_1bil.txt')
    # fix_file('vdc19_1bil.txt')


    to_latex()