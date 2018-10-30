##################################################################################################
# Plot t-SNE transformation
###################################################################################################

print('transform:')
Phi = clf.transform(X,zfile,np.arange(len(X)))
Phi = np.vstack((Phi[:,0],Phi[:,2])).transpose()
print('Phi:',Phi.shape,Phi)
# use t-SNE to visualize transformation
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects

proj = TSNE(random_state=42).fit_transform(Phi)

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("cividis", 2))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
#     plt.xlim(-25, 25)
#     plt.ylim(-25, 25)
    ax.axis('square')
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(2):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

scatter(proj,y)
plt.savefig('tsne_transformation.svg', dpi=120)
