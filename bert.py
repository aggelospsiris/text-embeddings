import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize texts and get BERT embeddings
def get_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
    return embeddings

embeddings = get_bert_embeddings(texts)

# Reduce dimensions for visualization using TSNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the embeddings
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100)

# Annotate points with text labels
for i, label in enumerate([f'text_{i}' for i in range(len(texts))]):
    plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=12, ha='right')

plt.title('Text Embeddings Visualization using BERT')
plt.show()