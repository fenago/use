import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# Load the Universal Sentence Encoder model
@st.cache(allow_output_mutation=True)
def load_model():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    return hub.load(module_url)

model = load_model()

def embed(input_text):
    return model(input_text)

# Streamlit app
st.title('Plagiarism Detector with Semantic Analysis')

# Text area for original text input
original_text = st.text_area("Enter the original text", "The quick brown fox jumps over the lazy dog.")

# Text area for rewritten text input
rewritten_text = st.text_area("Enter the rewritten text", "A fast auburn fox leaps over a dormant hound.")

# Function to plot and return the heatmap
def plot_heatmap(original_embeddings, rewritten_embeddings, original_text, rewritten_text):
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = sklearn_cosine_similarity(original_embeddings, rewritten_embeddings)
    sns.heatmap(corr, annot=True, cmap="YlGnBu", xticklabels=[rewritten_text], yticklabels=[original_text], ax=ax)
    return fig

# Function to plot and return the PCA visualization
def plot_pca(reduced_embeddings, labels):
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Sentences")
    ax.add_artist(legend1)
    return fig

if st.button('Analyze Similarity'):
    with st.spinner('Calculating embeddings...'):
        # Compute embeddings
        original_embedding = embed([original_text])
        rewritten_embedding = embed([rewritten_text])

        # Compute PCA
        all_embeddings = np.concatenate([original_embedding, rewritten_embedding])
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(all_embeddings)
        
        # Assign labels for PCA plot (1 for original, 2 for rewritten)
        labels = np.array([1, 2])

        # Plotting the heatmap
        heatmap_fig = plot_heatmap(original_embedding.numpy(), rewritten_embedding.numpy(), original_text, rewritten_text)
        st.pyplot(heatmap_fig)

        # Plotting the PCA visualization
        pca_fig = plot_pca(reduced_embeddings, labels)
        st.pyplot(pca_fig)
