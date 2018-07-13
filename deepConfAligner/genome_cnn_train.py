'''
Training code for implementing Deep Conflation Model using CNN encoder
'''
OMP_NUM_THREADS=8
import sys
import logging
import pickle
import random

from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import theano
import theano.tensor as tensor

from model.cnn_matching import init_params, init_tparams, build_model, build_encoder
from model.optimizers import Adam,Adagrad
from model.utils import unzip, zipp

from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy.random as ra

class mdict(dict):
    def __setitem__(self, key, value):
        """add the given value to the list of values for this key"""
        self.setdefault(key, []).append(value)


image_size = 30
image_h = 1
mutLength = int(image_size*0.3)


# # Creates the dataset
# def getDataset(display=True):
#     image_fname = "refseqImages_" + str(image_size) + ".txt"
#     data = []
#     with open(image_fname) as image_file:
#         for line in image_file:
#             d = eval(line)
#             data.append(d)
#     ref_image_data = {}
#     mut_image_data = mdict()
#     # image_label = []
#     for d in data:
#         if d[1] != 0:
#             key = d[0]
#             mut_image_data[key] = d[2]
#         else:
#             key = d[0]
#             ref_image_data[key] = d[2]
#     return mut_image_data,ref_image_data


# Creates the dataset
def getDataset(display=True):
    image_fname = "dataset/data_"+str(image_size)+"_"+str(mutLength)+".txt"
    print(image_fname)
    print(image_fname)
    modif = []
    actual = []
    modif2 = []
    actual2 = []
    with open(image_fname) as image_file:
        for line in image_file:
            d = eval(line)
            modif.append(d[1])
            actual.append(d[0])
    # for i in range(int(len(modif) / 100)):
    #     for j in range(int(len(modif) / 100)):
    #         modif2.append(modif[i + j * 100])
    #         actual2.append(actual[i + j * 100])
    # ref_image_data = {}
    # mut_image_data = mdict()
    # # image_label = []
    # for d in data:
    #     if d[1] != 0:
    #         key = d[0]
    #         mut_image_data[key] = d[2]
    #     else:
    #         key = d[0]
    #         ref_image_data[key] = d[2]
    return actual, modif


def prepare_data(seqs_x, max_len, n_chars, filter_h):
    pad = filter_h - 1
    x = []
    for rev in seqs_x:
        xx = []
        for i in range(pad):
            # we need pad the special <pad_zero> token.
            xx.append(n_chars - 1)
        for idx in rev:
            xx.append(idx)
        while len(xx) < max_len + 2 * pad:
            # we need pad the special <pad_zero> token.
            xx.append(n_chars - 1)
        x.append(xx)
    x = np.array(x, dtype='int32')
    return x

#building model
# max_len = image_size+1
# n_chars = 6
# img_w=24
# filter_hs=[2, 3, 4]
# feature_maps=100
# max_epochs = 1
# gamma = 10
# ncon = 50
# lrate = 0.0002,
# batch_size = 10
# dispFreq = 1
# validFreq = 10
# saveto='genome.npz'
#
# img_h = max_len + 2 * (filter_hs[-1] - 1)
#
# model_options = {}
# model_options['n_chars'] = n_chars
# model_options['img_w'] = img_w
# model_options['img_h'] = img_h
# model_options['feature_maps'] = feature_maps
# model_options['filter_hs'] = filter_hs
# model_options['max_epochs'] = max_epochs
# model_options['gamma'] = gamma
# model_options['ncon'] = ncon
# model_options['lrate'] = lrate
# model_options['batch_size'] = batch_size
# model_options['dispFreq'] = dispFreq
# model_options['validFreq'] = validFreq
# model_options['saveto'] = saveto
#
# # logger.info('Model options {}'.format(model_options))
#
# # logger.info('Building model...')
#
# filter_w = img_w
# filter_shapes = []
# pool_sizes = []
# for filter_h in filter_hs:
#     filter_shapes.append((feature_maps, 1, filter_h, filter_w))
#     pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))
#
# model_options['filter_shapes'] = filter_shapes
# model_options['pool_sizes'] = pool_sizes
#
# params = init_params(model_options)
# tparams = init_tparams(params)
#
# use_noise, inps, cost = build_model(tparams, model_options)
#
# # logger.info('Building encoder...')
# inps_e, feat_x, feat_y = build_encoder(tparams, model_options)
#
# # logger.info('Building functions...')
# f_emb = theano.function(inps_e, [feat_x, feat_y], name='f_emb')
#
# lr = tensor.scalar(name='lr')
# f_grad_shared, f_update = Adam(tparams, cost, inps, lr)


plt.ion()

figure, ax = plt.subplots(3,1,sharex=True,sharey=False)

lines, = ax[0].plot([],[], 'k-')
lines2, = ax[1].plot([],[], 'r-')
lines3, = ax[2].plot([],[], 'b-')


ax[0].set_autoscaley_on(True)
ax[0].set_autoscalex_on(True)
ax[1].set_autoscaley_on(True)
ax[1].set_autoscalex_on(True)
ax[2].set_autoscaley_on(True)
ax[2].set_autoscalex_on(True)
xdata = []
x3data = []
ydata = []
y2data = []
y3data = []

ctr = 0
# building model

# trainer

# trainer
def trainer(train, valid, test, n_chars=33, img_w=128, max_len=27, feature_maps=100,
            filter_hs=[2, 3, 4], max_epochs=20, gamma=10, ncon=100, lrate=0.0002,
            batch_size=100, dispFreq=10, validFreq=10, saveto='example.npz'):
    """ train, valid, test : datasets
        n_chars : vocabulary size
        img_w : character embedding dimension.
        max_len : the maximum length of a sentence
        feature_maps : the number of feature maps we used
        filter_hs: the filter window sizes we used
        max_epochs : The maximum number of epoch to run
        gamma: hyper-parameter using in ranking
        ncon: the number of negative samples we used for each postive sample
        lrate : learning rate
        batch_size : batch size during training
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation rank score after this number of update.
        saveto: where to save the result.
    """
    global ctr
    img_h = max_len + 2 * (filter_hs[-1] - 1)

    model_options = {}
    model_options['n_chars'] = n_chars
    model_options['img_w'] = img_w
    model_options['img_h'] = img_h
    model_options['feature_maps'] = feature_maps
    model_options['filter_hs'] = filter_hs
    model_options['max_epochs'] = max_epochs
    model_options['gamma'] = gamma
    model_options['ncon'] = ncon
    model_options['lrate'] = lrate
    model_options['batch_size'] = batch_size
    model_options['dispFreq'] = dispFreq
    model_options['validFreq'] = validFreq
    model_options['saveto'] = saveto

    logger.info('Model options {}'.format(model_options))

    logger.info('Building model...')

    filter_w = img_w
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))

    model_options['filter_shapes'] = filter_shapes
    model_options['pool_sizes'] = pool_sizes

    params = init_params(model_options)
    tparams = init_tparams(params)

    use_noise, inps, cost = build_model(tparams, model_options)

    logger.info('Building encoder...')
    inps_e, feat_x, feat_y = build_encoder(tparams, model_options)

    logger.info('Building functions...')
    f_emb = theano.function(inps_e, [feat_x, feat_y], name='f_emb')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = Adam(tparams, cost, inps, lr)

    logger.info('Training model...')

    uidx = 0
    seed = 1234
    curr = 0
    history_errs = []

    valid_x = prepare_data(valid[0], max_len, n_chars, filter_hs[-1])
    valid_y = prepare_data(valid[1], max_len, n_chars, filter_hs[-1])

    test_x = prepare_data(test[0], max_len, n_chars, filter_hs[-1])
    test_y = prepare_data(test[1], max_len, n_chars, filter_hs[-1])

    zero_vec_tensor = tensor.vector()
    zero_vec = np.zeros(img_w).astype(theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], updates=[
        (tparams['Wemb'], tensor.set_subtensor(tparams['Wemb'][n_chars - 1, :], zero_vec_tensor))])

    # Main loop
    for eidx in range(max_epochs):
        print("epoch {} ".format(eidx))

        prng = RandomState(seed - eidx - 1)

        trainA = train[0]
        trainB = train[1]

        num_samples = len(trainA)

        inds = np.arange(num_samples)
        prng.shuffle(inds)
        numbatches = len(inds) / batch_size
        for minibatch in range(numbatches):
            print("minibatch : ", minibatch)

            use_noise.set_value(0.)
            uidx += 1
            conprng = RandomState(seed + uidx + 1)

            x = [trainA[seq] for seq in inds[minibatch::numbatches]]
            y = [trainB[seq] for seq in inds[minibatch::numbatches]]

            cinds = conprng.random_integers(low=0, high=num_samples - 1, size=ncon * len(x))
            cy = [trainB[seq] for seq in cinds]

            x = prepare_data(x, max_len, n_chars, filter_hs[-1])
            y = prepare_data(y, max_len, n_chars, filter_hs[-1])
            cy = prepare_data(cy, max_len, n_chars, filter_hs[-1])

            feats_x, feats_y = f_emb(x, y)
            (r1, r3, r10, medr, meanr, h_meanr) = rank(feats_x, feats_y)

            cost = f_grad_shared(x, y, cy)
            print("cost {},r {}".format(cost, r1))
            f_update(lrate)

            xdata.append(ctr)
            ctr = ctr + 1
            ydata.append(cost)
            y2data.append(r1)

            lines.set_xdata(xdata)
            lines.set_ydata(ydata)
            lines2.set_xdata(xdata)
            lines2.set_ydata(y2data)
            # Need both of these in order to rescale
            ax[0].relim()
            ax[0].autoscale_view()
            ax[1].relim()
            ax[1].autoscale_view()
            # We need to draw *and* flush
            figure.canvas.draw()
            figure.canvas.flush_events()

            # the special token does not need to update.
            set_zero(zero_vec)

            if np.mod(uidx, dispFreq) == 0:
                logger.info('Epoch {} Update {} Cost {}'.format(eidx, uidx, cost))

            if np.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                logger.info('Computing ranks...')

                # valid_y,slocs = shuffle_valid(valid_y)

                feats_x, feats_y = f_emb(valid_x, valid_y)
                # (r1, r3, r10, medr, meanr, h_meanr) = rank_valid(feats_x, feats_y,slocs)
                (r1, r3, r10, medr, meanr, h_meanr) = rank(feats_x, feats_y)

                x3data.append(ctr)
                y3data.append(r1)

                lines3.set_xdata(x3data)
                lines3.set_ydata(y3data)

                ax[2].relim()
                ax[2].autoscale_view()
                # We need to draw *and* flush
                figure.canvas.draw()
                figure.canvas.flush_events()

                history_errs.append([r1, r3, r10, medr, meanr, h_meanr])

                logger.info('Valid Rank:{}, {}, {}, {},{},{}'.format(r1, r3, r10, medr, meanr, h_meanr))
                print('Valid Rank:{}, {}, {}, {},{},{}'.format(r1, r3, r10, medr, meanr, h_meanr))


                currscore = r1 + r3 + r10
                if currscore > curr:
                    curr = currscore
                    logger.info('Saving...')
                    params = unzip(tparams)
                    np.savez(saveto, history_errs=history_errs, **params)
                    logger.info('Done...')

    use_noise.set_value(0.)
    zipp(params, tparams)
    logger.info('Final results...')

    feats_x, feats_y = f_emb(valid_x, valid_y)
    (r1, r3, r10, medr, meanr, h_meanr) = rank(feats_x, feats_y)
    logger.info('Valid Rank:{}, {}, {}, {},{},{}'.format(r1, r3, r10, medr, meanr, h_meanr))

    feats_x, feats_y = f_emb(test_x, test_y)
    (r1, r3, r10, medr, meanr, h_meanr) = rank(feats_x, feats_y)
    logger.info('Test Rank:{}, {}, {}, {},{},{}'.format(r1, r3, r10, medr, meanr, h_meanr))

    # np.savez("./cnn_feats.npz", feats_x=feats_x, feats_y=feats_y)

    return (r1, r3, r10, medr, meanr, h_meanr)

def shuffle_valid(valid_y):
    x = np.arange(len(valid_y))
    # start_idx=0
    # r = ra.randint(0,len(valid_y))
    # # ra.shuffle(x[start_idx], lambda: r)
    # ra.shuffle(valid_y[start_idx], lambda: r)
    # # X = np.asarray(x[start_idx][0:batch_size])
    valid_y, x = shuffle(valid_y, x, random_state=0)
    return valid_y,x




# def trainer(train, valid, test, n_chars=6, img_w=24, max_len=150, feature_maps=25,
#             filter_hs=[2, 3, 4], max_epochs=1, gamma=10, ncon=50, lrate=0.0002,
#             batch_size=10, dispFreq=1, validFreq=2, saveto='10maygenome.npz'):
#     """ train, valid, test : datasets
#         n_chars : vocabulary size
#         img_w : character embedding dimension.
#         max_len : the maximum length of a sentence
#         feature_maps : the number of feature maps we used
#         filter_hs: the filter window sizes we used
#         max_epochs : The maximum number of epoch to run
#         gamma: hyper-parameter using in ranking
#         ncon: the number of negative samples we used for each postive sample
#         lrate : learning rate
#         batch_size : batch size during training
#         dispFreq : Display to stdout the training progress every N updates
#         validFreq : Compute the validation rank score after this number of update.
#         saveto: where to save the result.
#     """
#     logger.info('Model options {}'.format(model_options))
#     global ctr
#     logger.info('Training model...')
#
#     uidx = 0
#     seed = 1234
#     curr = 0
#     history_errs = []
#
#
#
#     valid_x = prepare_data(valid[0][0:10], max_len, n_chars, filter_hs[-1])
#     valid_y = prepare_data(valid[1][0:10], max_len, n_chars, filter_hs[-1])
#
#     valid_y,slocs = shuffle_valid(valid_y)
#     print("shuffled locs : ",slocs)
#
#     # print("validation data size : ", len(valid_x))
#     # print(test[0][:10])
#     # print(test[1][:10])
#
#     test_x = prepare_data(test[0], max_len, n_chars, filter_hs[-1])
#     test_y = prepare_data(test[1], max_len, n_chars, filter_hs[-1])
#
#     zero_vec_tensor = tensor.vector()
#     zero_vec = np.zeros(img_w).astype(theano.config.floatX)
#     set_zero = theano.function([zero_vec_tensor], updates=[
#         (tparams['Wemb'], tensor.set_subtensor(tparams['Wemb'][n_chars - 1, :], zero_vec_tensor))])
#
#     print("max epochs : ", max_epochs)
#     # Main loop
#     for eidx in range(max_epochs):
#         prng = RandomState(seed - eidx - 1)
#
#         trainA = train[0]
#         trainB = train[1]
#
#         num_samples = len(trainA)
#
#         inds = np.arange(num_samples)
#         prng.shuffle(inds)
#         numbatches = len(inds) / batch_size
#
#         print("epoch {} ".format(eidx))
#         print(numbatches)
#
#         for minibatch in range(numbatches):
#             print("minibatch : ", minibatch)
#             use_noise.set_value(0.)
#             uidx += 1
#             conprng = RandomState(seed + uidx + 1)
#
#             x = [trainA[seq] for seq in inds[minibatch::numbatches]]
#             y = [trainB[seq] for seq in inds[minibatch::numbatches]]
#
#
#             # actual_loc = [iloc for iloc in range(batch_size)]
#             # r = random.random()
#             # random.shuffle(y[0], lambda: r)
#             # random.shuffle(actual_loc[0], lambda: r)
#
#             # print(x)
#             # print(y)
#             # print("num of elements in a batch ",len(x))
#
#             cinds = conprng.random_integers(low=0, high=num_samples - 1, size=ncon * len(x))
#             cy = [trainB[seq] for seq in cinds]
#
#             x = prepare_data(x, max_len, n_chars, filter_hs[-1])
#             y = prepare_data(y, max_len, n_chars, filter_hs[-1])
#             cy = prepare_data(cy, max_len, n_chars, filter_hs[-1])
#
#             # print(x)
#             # print(y)
#
#             feats_x, feats_y = f_emb(x, y)
#             (r1, r3, r10, medr, meanr, h_meanr) = rank(feats_x, feats_y)
#
#             cost = f_grad_shared(x, y, cy)
#             print("cost {},r {}".format(cost,r1))
#             f_update(lrate)
#
#             xdata.append(ctr)
#             ctr =ctr+1
#             ydata.append(cost)
#             y2data.append(r1)
#
#             lines.set_xdata(xdata)
#             lines.set_ydata(ydata)
#             lines2.set_xdata(xdata)
#             lines2.set_ydata(y2data)
#             # Need both of these in order to rescale
#             ax[0].relim()
#             ax[0].autoscale_view()
#             ax[1].relim()
#             ax[1].autoscale_view()
#             # We need to draw *and* flush
#             figure.canvas.draw()
#             figure.canvas.flush_events()
#
#             # the special token does not need to update.
#             set_zero(zero_vec)
#
#             if np.mod(uidx, dispFreq) == 0:
#                 logger.info('Epoch {} Update {} Cost {}'.format(eidx, uidx, cost))
#
#             if np.mod(uidx, validFreq) == 0:
#                 # print("validating")
#                 use_noise.set_value(0.)
#                 logger.info('Computing ranks...')
#
#                 # v_k = int((uidx%len(valid_x))*100)
#
#                 valid_x = x
#                 valid_y = y
#                 valid_y,slocs = shuffle_valid(valid_y)
#
#                 # feats_x, feats_y = f_emb(valid_x[v_k:v_k+99], valid_y[v_k:v_k+99])
#                 feats_x, feats_y = f_emb(valid_x, valid_y)
#                 (r1, r3, r10, medr, meanr, h_meanr) = rank_valid(feats_x, feats_y,slocs)
#
#                 x3data.append(ctr)
#                 y3data.append(r1)
#
#                 lines3.set_xdata(x3data)
#                 lines3.set_ydata(y3data)
#
#                 ax[2].relim()
#                 ax[2].autoscale_view()
#                 # We need to draw *and* flush
#                 figure.canvas.draw()
#                 figure.canvas.flush_events()
#
#                 history_errs.append([r1, r3, r10, medr, meanr, h_meanr])
#
#                 logger.info('Valid Rank:{}, {}, {}, {},{},{}'.format(r1, r3, r10, medr, meanr, h_meanr))
#                 print('Valid Rank:{}, {}, {}, {},{},{}'.format(r1, r3, r10, medr, meanr, h_meanr))
#
#                 currscore = r1 + r3 + r10
#                 if currscore > curr:
#                     curr = currscore
#                     logger.info('Saving...')
#                     params = unzip(tparams)
#                     np.savez(saveto, history_errs=history_errs, **params)
#                     logger.info('Done...')
#
#     use_noise.set_value(0.)
#     zipp(params, tparams)
#     logger.info('Final results...')
#
#     feats_x, feats_y = f_emb(valid_x, valid_y)
#     (r1, r3, r10, medr, meanr, h_meanr) = rank(feats_x, feats_y)
#     logger.info('Valid Rank:{}, {}, {}, {},{},{}'.format(r1, r3, r10, medr, meanr, h_meanr))
#
#     feats_x, feats_y = f_emb(test_x, test_y)
#
#     (r1, r3, r10, medr, meanr, h_meanr) = rank(feats_x, feats_y)
#     logger.info('Test Rank:{}, {}, {}, {},{},{}'.format(r1, r3, r10, medr, meanr, h_meanr))
#
#     # np.savez("./cnn_feats.npz", feats_x=feats_x, feats_y=feats_y)
#
#     return (r1, r3, r10, medr, meanr, h_meanr)

def rank_valid(x, y,locs):
    """ x,y: (n_samples, n_feats)
    """
    # print(x)
    # print(y)
    npts = x.shape[0]
    n_feats = x.shape[1]

    index_list = []

    print(x,y)

    ranks = np.zeros(npts)
    for index in range(npts):
        # Get query text
        im = x[index].reshape(1, n_feats)
        # print(index)
        # Compute scores
        d = np.dot(im, y.T).flatten()
        print(d)
        inds = np.argsort(d)[::-1]
        # print(inds[0])
        index_list.append(inds[0])
        # print(index_list)

        ranks[index] = np.where(inds == locs[index])[0][0]

    # Compute metrics
    print("indices : ",index_list)
    print("ranks : ",ranks)
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r3 = 100.0 * len(np.where(ranks < 3)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = np.mean(ranks) + 1
    h_meanr = 1. / np.mean(1. / (ranks + 1))
    return (r1, r3, r10, medr, meanr, h_meanr)


def rank(x, y):
    """ x,y: (n_samples, n_feats)
    """
    npts = x.shape[0]
    n_feats = x.shape[1]

    index_list = []

    ranks = np.zeros(npts)
    for index in range(npts):
        # Get query text
        im = x[index].reshape(1, n_feats)

        # Compute scores
        d = np.dot(im, y.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r3 = 100.0 * len(np.where(ranks < 3)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = np.mean(ranks) + 1
    h_meanr = 1. / np.mean(1. / (ranks + 1))
    return (r1, r3, r10, medr, meanr, h_meanr)


def create_valid(train_set, valid_portion=0.10):
    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)

    return train, valid


if __name__ == "__main__":

    # using predefined split or doing 10-fold cross validation
    data = "cross_validation" #sys.argv[1]
    # using the correct name to query the wrong name or reverse
    query = "normal" #sys.argv[2]

    # https://docs.python.org/2/howto/logging-cookbook.html
    logger = logging.getLogger('30mereval_cnn_{}_{}'.format(data, query))
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('30mereval_cnn_{}_{}.log'.format(data, query))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)

    saveto = "30mercnn_results_{}_{}".format(data, query)

    if data == "predefined_split":

        x = pickle.load(open("./data/Genome_data_predefined_split.p", "rb"))
        train, valid, test, char2ix, ix2char = x[0], x[1], x[2], x[3], x[4]
        del x

        text = train[0] + train[1] + valid[0] + valid[1] + test[0] + test[1]
        length = []
        for sent in text:
            length.append(len(sent))

        max_len = np.max(length) + 1

        n_chars = len(ix2char)

        ix2char[n_chars] = '<pad_zero>'
        char2ix['<pad_zero>'] = n_chars
        n_chars = n_chars + 1

        if query == "reverse":
            trainA, trainB = train[0], train[1]
            validA, validB = valid[0], valid[1]
            testA, testB = test[0], test[1]

            train = [trainB, trainA]
            valid = [validB, validA]
            test = [testB, testA]

        (r1, r3, r10, medr, meanr, h_meanr) = trainer(train, valid, test,
                                                      n_chars=n_chars, max_len=max_len, saveto=saveto)

    elif data == "cross_validation":

        x = pickle.load(open("./data/Genome30_data.p", "rb"))
        text, char2ix, ix2char = x[0], x[1], x[2]
        del x

        text0 = text[0] + text[1]
        length = []
        for sent in text0:
            length.append(len(sent))

        max_len = np.max(length) + 1

        n_chars = len(ix2char)

        ix2char[n_chars] = '<pad_zero>'
        char2ix['<pad_zero>'] = n_chars
        n_chars = n_chars + 1

        if query == "normal":
            textA = text[0]
            textB = text[1]
        elif query == "reverse":
            textA = text[1]
            textB = text[0]

        results = []
        i = 0
        kf = KFold(len(textA), n_folds=10, random_state=1234)
        for train_index, test_index in kf:
            train_index = train_index.tolist()
            test_index = test_index.tolist()

            train_x = [textA[ix] for ix in train_index]
            train_y = [textB[ix] for ix in train_index]

            test_x = [textA[ix] for ix in test_index]
            test_y = [textB[ix] for ix in test_index]

            train = (train_x, train_y)
            test = (test_x, test_y)
            train, valid = create_valid(train, valid_portion=0.10)
            (r1, r3, r10, medr, meanr, h_meanr) = trainer(train, valid, test,
                                                          n_chars=n_chars, max_len=max_len, saveto=saveto)

            logger.info('cv: {} test rank: {}, {}, {}, {},{},{}'.format(i, r1, r3, r10, medr, meanr, h_meanr))
            i = i + 1
            results.append([r1, r3, r10, medr, meanr, h_meanr])

        np.savez("./30mercnn_ten_fold_{}_{}.npz".format(data, query), results=results)

