import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

np.random.seed(123)

# dictionary beta, n_tw

FILE_NAME = './ps4_data/abstract_nips21_NIPS2008_0517.txt.ready'

NUM_TOPIC = 0
ALPHA = []
BETA = {}
dic_id = {}
id_dic = {}
doc = []
count = {}

def read_file(f):
    with open(f, 'rb') as f_des:
        line = f_des.readline()
        NUM_TOPIC = int(line.replace('\n', ''))

        line = f_des.readline()
        ALPHA = [float(i) for i in line.replace('\n', '').split()]

        BETA = {}
        lines = f_des.readlines()

        idx = 0
        for line in lines:
            line = line.replace('\n', '').split()
            word = line[0]
            if word in dic_id.keys():  
                count[dic_id[word]] += 1
            else:
                dic_id[word] = idx
                id_dic[idx] = word
                idx += 1
                count[dic_id[word]] = 1
                assert NUM_TOPIC == len(line[1:])
                BETA[dic_id[word]] = [float(i) for i in line[1:]]

            doc.append(dic_id[word])

    return NUM_TOPIC, ALPHA, BETA, dic_id, id_dic, doc, count

def collapsed_gibbs_sampler(iter, alpha, beta):
    # iter: number of iterations
    theta = np.random.dirichlet(alpha)
    z = np.random.multinomial(len(doc), theta)

    n_t = np.zeros(len(alpha))
    n_tw = np.zeros([len(alpha), len(dic_id)])
    for i in range(len(alpha)):
        n_t[i] = np.sum(z == i)
    for i in range(len(alpha)):
        for j in id_dic:
            n_tw[i][j] = np.sum(np.logical_and(doc == j, z == i))

    res= {}
    sample_order = np.arange(len(doc))
    for ite in xrange(iter):
        t_res = np.zeros(len(doc))
        np.random.shuffle(sample_order)
        for i in sample_order:
            # p_val
            p_val = np.zeros(len(alpha))
            n_t[z[i]] -= 1
            for k in range(len(alpha)):
                p_val[k] = (n_t[k] + alpha[k]) * beta[doc[i]][k]
            # print p_val
            p_val[k] /= np.sum(p_val)
            p_val = np.cumsum(p_val)
            z_i = (p_val > np.random.uniform()).argmax() # z_i = np.random.multinomial(1, p_val)

            n_t[z_i] += 1
            t_res[i] = z_i
        res[ite] = t_res
    return res

def main():

    NUM_TOPIC, ALPHA, BETA, dic_id, id_dic, doc, count = read_file(FILE_NAME)

    print len(BETA), len(dic_id), len(id_dic), len(doc), len(count)

    ite = 1e4
    samples = collapsed_gibbs_sampler(int(ite), ALPHA, BETA)

    post_theta = np.zeros(len(ALPHA))
    for i in samples.keys()[50:]:
        for j in range(len(doc)):
            post_theta[int(samples[i][j])] += 1
    normalizer = ite * (np.sum(ALPHA) + len(doc))
    post_theta += ite * np.array(ALPHA)
    post_theta /= normalizer

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(post_theta)
    plt.show()


    print np.sum(post_theta)





if __name__ == "__main__":
    main()