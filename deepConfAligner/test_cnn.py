
import numpy as np
import theano

from model.cnn_matching import init_params, init_tparams, build_model, build_encoder, load_params

import matplotlib.pyplot as plt
import dataGenerator as dg

from sklearn.metrics import confusion_matrix
import pickle
import itertools

image_size = 100

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



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


data = "cross_validation"  # sys.argv[1]
query = "normal"  # sys.argv[2]
saveto = "23maycnn_results_{}_{}".format(data, query)

b= np.load(saveto+".npz")

max_len = image_size+1
n_chars = 4
img_w=128
filter_hs=[2, 3, 4]
feature_maps=100
max_epochs = 1
gamma = 10
ncon = 50
lrate = 0.0002,
dispFreq = 100
validFreq = 81
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
model_options['dispFreq'] = dispFreq
model_options['validFreq'] = validFreq
model_options['saveto'] = saveto

# logger.info('Model options {}'.format(model_options))

# logger.info('Building model...')

filter_w = img_w
filter_shapes = []
pool_sizes = []
for filter_h in filter_hs:
    filter_shapes.append((feature_maps, 1, filter_h, filter_w))
    pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))

model_options['filter_shapes'] = filter_shapes
model_options['pool_sizes'] = pool_sizes

params = load_params(model_options,b)
tparams = init_tparams(params)

# use_noise, inps, cost = build_model(tparams, model_options)

# logger.info('Building encoder...')
inps_e, feat_x, feat_y = build_encoder(tparams, model_options)

# logger.info('Building functions...')
f_emb = theano.function(inps_e, [feat_x, feat_y], name='f_emb')

print("model loaded")



def confusionmatrixlist(x,y,l,cl):
    n_feats = x.shape[1]

    im = x.reshape(1, n_feats)

    d = np.dot(im, y.T).flatten()
    inds = np.argsort(d)[::-1]

    dummy = np.zeros(len(inds))
    dummy[inds[0]] = 1

    cl.append(list(dummy))
    return cl


def acc(x, y,l):
    """ x,y: (n_samples, n_feats)
    """
    # print(x)
    # print(y)
    n_feats = x.shape[1]

    # Get query text
    im = x.reshape(1, n_feats)
    # print(index,im)
    # Compute scores
    d = np.dot(im, y.T).flatten()
    # print(d)
    inds = np.argsort(d)[::-1]
    if(l[inds[0]] == 1):
        return 1
    else:
        return 0

index_loc = {}

set,labels=dg.readRealData()

max_len = image_size + 1




plt.ion()

figure, ax = plt.subplots(1,1,sharex=True,sharey=False)

lines, = ax.plot([],[], 'k-')

ax.set_ylim([0,100])
# ax.set_autoscaley_on(True)
ax.set_autoscalex_on(True)

xdata = []
ydata = []

accuracy=0
aclist = []
count = 0
tacc = 0

for i in range(len(set)):
    textAi = [set[i][0]]
    textBi = set[i][1]
    labeli = labels[i][0]

    test_x = prepare_data(textAi, max_len, n_chars, filter_hs[-1])
    test_y = prepare_data(textBi, max_len, n_chars, filter_hs[-1])

    feats_x, feats_y = f_emb(test_x, test_y)

    # (r1, r3, r10, medr, meanr, h_meanr) = locrank(feats_x, feats_y)
    accuracy += acc(feats_x,feats_y,labeli)
    aclist = confusionmatrixlist(feats_x,feats_y,labeli,aclist)



    if (i+1)%100 == 0:
        xdata.append(i)
        ydata.append(accuracy)
        print(accuracy)

        lines.set_xdata(xdata)
        lines.set_ydata(ydata)

        ax.relim()
        ax.autoscale_view()

        figure.canvas.draw()
        figure.canvas.flush_events()

        tacc = tacc+accuracy
        accuracy=0
        count = count+1


file = open("100mertestslabels","wb")
pickle.dump(labels,file)
file2 = open("100mertestspredicted","wb")
pickle.dump(aclist,file2)
file.close()
file2.close()

labels = pickle.load(open("100mertestslabels","rb"))
aclist = pickle.load(open("100mertestspredicted","rb"))


cnf_matrix = confusion_matrix((np.reshape(np.hstack(labels),(-1),)),np.reshape(np.hstack(aclist),(-1)))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Matched', 'Unmatched'],
                      title='Confusion matrix, with normalization')
tn, fp, fn, tp = cnf_matrix.ravel().astype(float)
recall = tp/(tp+fn)
precision = tp/(tp +fp)
print ("Accuracy ", (tp+tn)/(tn+fp+fn+tp))
print("Precision ", (tp/(tp +fp)))
print("Recall ", (tp/(tp+fn)))
print ("F1 Score ",(precision*recall)/(precision+recall))

figure.canvas.draw()
figure.canvas.flush_events()
plt.show()

print("mean accuracy ",(tacc/count))
