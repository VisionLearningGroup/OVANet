import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
args = sys.argv
N = 11 # Number of labels
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired
colorst = [colormap(i) for i in np.linspace(0, 0.9, N)]
def plot_embedding(X, label, title=None):
    plt.figure()
    ax = plt.subplot(111)
    #color_dict = {0:'b',1:'r'}
    for i in range(N):
        inds = np.where(label==i)[0]
        plt.scatter(X[inds, 0], X[inds, 1],color=colorst[i],label=str(i))

    ax.tick_params(labelbottom="off", bottom="off")
    ax.tick_params(labelleft="off", left="off")
    plt.tick_params(color='white')
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    #ax.legend(loc='upper left')
    #plt.legend()
    plt.savefig(title)
    return X#(X - x_min) / (x_max - x_min)
def plot_embedding2(X, label, title=None):
    plt.figure()
    ax = plt.subplot(111)
    color_dict = {0:'b',1:'r'}
    inds2 = np.where(label==2)[0]
    inds1 = np.where(label==1)[0]
    inds0 = np.where(label==0)[0]
    plt.scatter(X[inds0, 0], X[inds0, 1],color='r',alpha=0.1)
    #inds = np.where(label == 1)[0]
    plt.scatter(X[inds1, 0], X[inds1, 1], color='b', alpha=0.1)
    plt.scatter(X[inds2, 0], X[inds2, 1], color='g', alpha=0.1)
    ax.tick_params(labelbottom="off", bottom="off")
    ax.tick_params(labelleft="off", left="off")
    plt.tick_params(color='white')
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    #ax.legend(loc='upper left')
    #plt.legend()
    plt.savefig(title)
    return X  # (X - x_min) / (x_max - x_min)
X_t = np.load(args[1])#[:,:500]
rand_v = np.random.permutation(X_t.shape[0])
#X_t = X_t[rand_v[:5000]]
label_t = np.load(args[3])
#label_t = label_t[rand_v[:5000]]
X_s = np.load(args[2])#[:,:500]
rand_v = np.random.permutation(X_s.shape[0])
X_s = X_s[rand_v[:2000]]
label_s = np.load(args[4])
label_s = label_s[rand_v[:2000]]
X_t = X_t.reshape(X_t.shape[0],X_t.shape[1])
print(label_t.shape)
print(label_s.shape)

X = normalize(np.r_[X_s,X_t])
Y = np.r_[label_s,label_t]
Y = Y.reshape(Y.shape[0])

X_embedded = TSNE(n_components=2, perplexity=30).fit_transform(X)
x_min, x_max = np.min(X_embedded, 0), np.max(X_embedded, 0)
X_embedded = (X_embedded - x_min) / (x_max - x_min)
X_scaled = plot_embedding(X_embedded[:X_s.shape[0]],Y[:X_s.shape[0]],title=args[5]+"_source")
X_scaled = plot_embedding(X_embedded[X_s.shape[0]:],Y[X_s.shape[0]:],title=args[5]+"_target")
inds_unk = np.where(Y==10)[0]

Y[:X_s.shape[0]] = 0
Y[X_s.shape[0]:] = 1
Y[inds_unk] = 2
X_scaled = plot_embedding2(X_embedded,Y,title=args[5]+'2')
np.save('embed.npy',X_scaled)
