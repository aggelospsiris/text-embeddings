import re
import math
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from collections import defaultdict
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Download the stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# List of texts
texts = [
    "Technology has become an integral part of our daily lives, influencing everything from how we communicate to how we work and entertain ourselves. Innovations in artificial intelligence, blockchain, and the Internet of Things are driving the next wave of technological advancements, promising to transform industries and create new opportunities for growth and efficiency.",
    "Artificial intelligence (AI) is rapidly advancing, with applications ranging from autonomous vehicles to personalized recommendations and predictive analytics. While AI holds great promise for improving efficiency and solving complex problems, it also raises ethical questions about bias, accountability, and the impact on employment.",
    "The healthcare industry is undergoing significant transformations with advancements in medical research, technology, and patient care. Innovations such as telemedicine, personalized medicine, and robotic surgery are enhancing the quality and accessibility of healthcare services. Preventive care and holistic approaches are gaining traction, emphasizing the importance of lifestyle choices in maintaining health and well-being.",
    "Mental health is a critical aspect of overall well-being, affecting how individuals think, feel, and interact with others. Awareness and understanding of mental health issues have increased, leading to more open discussions and reduced stigma. Access to mental health services, early intervention, and supportive environments are essential for promoting mental well-being.",
    "Climate change poses one of the greatest challenges of our time, impacting ecosystems, weather patterns, and human societies globally. Sustainable practices, such as reducing carbon emissions, conserving natural resources, and transitioning to renewable energy sources, are essential to mitigate these effects.",
    "The transition to renewable energy sources, such as solar, wind, and hydroelectric power, is essential for combating climate change and reducing dependence on fossil fuels. The green economy, which focuses on sustainable development and environmental protection, offers significant opportunities for innovation, job creation, and economic growth.",
    "Globalization has interconnected economies, cultures, and societies, fostering greater cultural exchange and understanding. While this has led to economic growth and innovation, it has also highlighted disparities and challenges, such as labor exploitation and cultural homogenization.",
    "Social media has revolutionized the way we communicate, share information, and connect with others. Platforms like Facebook, Twitter, and Instagram have become essential tools for personal expression, business marketing, and social activism. However, the rise of digital communication also brings challenges such as misinformation, privacy concerns, and the impact on mental health.",
    "The global economy is influenced by a myriad of factors, including technological advancements, geopolitical developments, and consumer behavior. Understanding these trends is crucial for businesses and policymakers to navigate market dynamics and make informed decisions. Economic indicators such as GDP growth, inflation rates, and employment figures provide valuable insights into the health of an economy.",
    "Education is the foundation of a prosperous society, empowering individuals with the knowledge and skills needed to succeed in an ever-changing world. The rise of digital learning platforms and online courses has democratized access to education, enabling people from diverse backgrounds to pursue lifelong learning opportunities."
]

# Function to preprocess text
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return words

# Create an inverted index
inverted_index = defaultdict(list)
doc_words = defaultdict(dict)
word_docs = defaultdict(dict)

for idx, text in enumerate(texts):
    words = preprocess(text)
    for word in words:
        if word not in inverted_index or idx not in inverted_index[word]:
            inverted_index[word].append(idx)
        if word not in word_docs:
            word_docs[word] = {}
        if idx not in word_docs[word]:
            word_docs[word][idx] = 0
        word_docs[word][idx] += 1
        if idx not in doc_words:
            doc_words[idx] = {}
        if word not in doc_words[idx]:
            doc_words[idx][word] = 0
        doc_words[idx][word] += 1

# Calculate DF for each word
word_df = {word: len(docs) for word, docs in word_docs.items()}
total_docs = len(texts)

# Calculate TF-IDF for each word in each document
tf_idf = defaultdict(dict)
for word, docs in word_docs.items():
    for doc_id, tf in docs.items():
        tf_idf[word][doc_id] = tf * math.log(total_docs / (1 + word_df[word]))

# Create the graph
G = nx.Graph()  # Use undirected graph for text nodes

# Add text nodes
for doc_id in range(total_docs):
    G.add_node(f'text_{doc_id}', type='text')

# Add edges between text nodes based on shared words
for doc1 in range(total_docs):
    for doc2 in range(doc1 + 1, total_docs):
        common_words = set(doc_words[doc1].keys()).intersection(set(doc_words[doc2].keys()))
        if not common_words:
            continue
        weight = sum(tf_idf[word][doc1] * tf_idf[word][doc2] for word in common_words)
        if weight > 0:
            G.add_edge(f'text_{doc1}', f'text_{doc2}', weight=weight)

# Mapping node identifiers to integer indices
node_mapping = {f'text_{i}': i for i in range(total_docs)}
edges = list(G.edges(data=True))
edge_index = torch.tensor([[node_mapping[e[0]], node_mapping[e[1]]] for e in edges], dtype=torch.long).t().contiguous()
edge_weight = torch.tensor([e[2]['weight'] for e in edges], dtype=torch.float)

x = torch.eye(total_docs)  # Use one-hot encoding for simplicity

data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(total_docs, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, total_docs, heads=1, concat=True, dropout=0.6)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x

# Train the GAT model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.x)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    embeddings = model(data).cpu().numpy()

# Reduce dimensions for visualization using TSNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the embeddings
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100)

# Annotate points with text labels
for i, label in enumerate([f'text_{i}' for i in range(total_docs)]):
    plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=12, ha='right')

plt.title('Text Node Embeddings Visualization using GAT')
plt.show()