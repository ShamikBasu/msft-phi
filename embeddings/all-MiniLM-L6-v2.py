from sentence_transformers import SentenceTransformer, util
import torch
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = list(model.encode(["I am shamik Basu","I am shamik Basu"]))

from sklearn.decomposition import PCA
PCA_model = PCA(n_components = 2)
PCA_model.fit(embeddings)
values = PCA_model.transform(embeddings)

import matplotlib.pyplot as plt
import mplcursors

sentences = ["I am shamik Basu","Shamik Basu I am"]

# Create a scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(values[:, 0], values[:, 1], color="blue", label="Sentences")

# Add title and labels
ax.set_title("2D Sentence Embeddings")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")

# Use mplcursors to display sentences on hover
cursor = mplcursors.cursor(scatter, hover=True)

# Customize the tooltip to show the actual sentence
@cursor.connect("add")
def on_add(sel):
    # Get the index of the selected point
    index = sel.index
    # Set the annotation text to the corresponding sentence
    sel.annotation.set_text(sentences[index])

# Show the plot
plt.legend()
plt.show()