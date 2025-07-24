import random
import math
import re
from collections import Counter

dialogue_memory = []
def pad_sequences(seq, max_len):
    return seq + [0] * (max_len - len(seq))
def text_to_sequences(tokens, vocab):
    return [vocab.get(word, 0) for word in tokens]
def softmax(x):
    e_x = [math.exp(i) for i in x]
    total = sum(e_x)
    return [i / total for i in e_x]
max_len = 10  # or however long your sequences are

# Preprocessing

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def tokenize(text):
    stopwords = {'i', 'the', 'a', 'an', 'is', 'in', 'on', 'of', 'and', 'to', 'with'}
    return [word for word in preprocess(text) if word not in stopwords]

# TF-IDF

def compute_tfidf(corpus):
    def compute_tf(text):
        tf_text = Counter(text)
        for i in tf_text:
            tf_text[i] = tf_text[i] / len(text)
        return tf_text

    def compute_idf(word, corpus):
        return math.log10(len(corpus) / sum([1 for text in corpus if word in text]))

    documents_list = [tokenize(text) for text in corpus]

    idf_dict = {}
    for doc in documents_list:
        for word in doc:
            if word not in idf_dict:
                idf_dict[word] = compute_idf(word, documents_list)

    tfidf_documents = []
    for doc in documents_list:
        tfidf_doc = {}
        tf_doc = compute_tf(doc)
        for word in tf_doc:
            tfidf_doc[word] = tf_doc[word] * idf_dict[word]
        tfidf_documents.append(tfidf_doc)

    return tfidf_documents

# Embedding

def build_embeddings(vocab, dim=10):
    return {word: [random.uniform(-1, 1) for _ in range(dim)] for word in vocab}

def embed_sequence(tokens, embeddings, dim):
    return [embeddings[word] if word in embeddings else [0.0]*dim for word in tokens]

# RNN-like structure
class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.last_inputs = []
        self.last_hs = []
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.Wxh = [[random.uniform(-1, 1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.Whh = [[random.uniform(-1, 1) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.Why = [[random.uniform(-1, 1) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.bh = [random.uniform(-1, 1) for _ in range(hidden_dim)]
        self.by = [random.uniform(-1, 1) for _ in range(output_dim)]
        self.h = [0.0] * hidden_dim
        
    def step(self, x):
        self.last_inputs.append(x)
        self.last_hs.append(self.h[:])  # store copy

        next_h = [0.0] * self.hidden_dim
        for i in range(self.hidden_dim):
            val = self.bh[i]
            for j in range(self.input_dim):
                val += x[j] * self.Wxh[j][i]
            for j in range(self.hidden_dim):
                val += self.h[j] * self.Whh[j][i]
            next_h[i] = math.tanh(val)
        self.h = next_h

        y = [0.0] * self.output_dim
        for i in range(self.output_dim):
            val = self.by[i]
            for j in range(self.hidden_dim):
                val += self.h[j] * self.Why[j][i]
            y[i] = val
        return y
        
    def reset(self):
        self.h = [0.0] * self.hidden_dim
    
    def backward(self, d_y, learn_rate=0.01):
        dWhy = [[0.0]*self.output_dim for _ in range(self.hidden_dim)]
        dby = d_y[:]
        dh = [0.0] * self.hidden_dim

        # ∂L/∂Why and ∂L/∂by
        for i in range(self.output_dim):
            for j in range(self.hidden_dim):
                dWhy[j][i] += self.h[j] * d_y[i]

        for j in range(self.hidden_dim):
            for i in range(self.output_dim):
                dh[j] += d_y[i] * self.Why[j][i]

        dWxh = [[0.0]*self.hidden_dim for _ in range(self.input_dim)]
        dWhh = [[0.0]*self.hidden_dim for _ in range(self.hidden_dim)]
        dbh = [0.0]*self.hidden_dim

        for t in reversed(range(len(self.last_inputs))):
            h = self.last_hs[t]
            x = self.last_inputs[t]

            dhraw = [(1 - h_i ** 2) * dh_i for h_i, dh_i in zip(h, dh)]  # tanh'

            for i in range(self.hidden_dim):
                dbh[i] += dhraw[i]
                for j in range(self.input_dim):
                    dWxh[j][i] += x[j] * dhraw[i]
                for j in range(self.hidden_dim):
                    dWhh[j][i] += self.last_hs[t-1][j] * dhraw[i] if t > 0 else 0.0

            dh = [sum(dhraw[i] * self.Whh[j][i] for i in range(self.hidden_dim)) for j in range(self.hidden_dim)]

        # Apply updates
        for i in range(self.hidden_dim):
            self.bh[i] -= learn_rate * dbh[i]
            for j in range(self.input_dim):
                self.Wxh[j][i] -= learn_rate * dWxh[j][i]
            for j in range(self.hidden_dim):
                self.Whh[j][i] -= learn_rate * dWhh[j][i]
        for i in range(self.output_dim):
            self.by[i] -= learn_rate * dby[i]
            for j in range(self.hidden_dim):
                self.Why[j][i] -= learn_rate * dWhy[j][i]

        # Reset memory
        self.last_inputs = []
        self.last_hs = []


# Sample phrases
phrases = [
    "I love coding.", "AI is fascinating.", "Natural Language Processing is complex.",
    "My name is Gamii", "Yes, will do!", "I'm sorry, I can't do that.",
    "Thank you very much!", "Sure I will do that!", "Hello, how can I assist you today?",
    "Good morning! What's on your agenda?", "I'm here to help you with anything you need.",
    "Let's get started with your task.", "I'm learning new things every day.",
    "Thank you for your patience.", "Great job!", "That's interesting, tell me more.",
    "I am programmed to assist you.", "Let's work together to solve this.",
    "I appreciate your feedback.", "How can I improve my responses?",
    "It's a beautiful day!", "I'm happy to help you.", "What can I do for you?",
    "Let's make today productive.", "I enjoy learning from our interactions.",
    "Have a great day!", "I am here to support you.", "Let's accomplish great things!",
    "I can provide information on various topics.", "What's your favorite activity?",
    "I am learning to understand you better.", "Thank you for chatting with me.",
    "I'm excited to assist you.", "Let's explore new ideas together.",
    "I am continually improving.", "How can I be more helpful?",
    "I strive to provide accurate information.", "I love coding and learning new things.",
    "Artificial Intelligence is fascinating.", "Natural Language Processing is complex.",
    "My name is Gamii.", "Yes, will do!", "I'm sorry, I can't do that.",
    "Thank you very much!", "Sure, I will do that.", "Let's dive deeper into this topic.",
    "I am here to assist you anytime.", "Your satisfaction is my goal.",
    "Let's work through this step-by-step.", "I am dedicated to helping you.",
    "What interests you the most?", "I am here to learn and grow with you.",
    "Let's tackle this challenge together.", "I am eager to learn from our conversations.",
    "Feel free to ask me anything.", "Your feedback is valuable to me.",
    "I am here to make your life easier.", "Let's discover something new today.",
    "I am constantly improving to assist you better.", "What would you like to know more about?",
    "I am programmed to help you achieve your goals.", "Let's take on this challenge together.",
    "I am here to support your endeavors.", "Your input helps me become better.",
    "Let's make this a productive session.", "I am happy to assist you with any task.",
    "What would you like to do next?", "I strive to be the best assistant for you.",
    "Thank you for using me, I appreciate it.", "I am here to provide valuable insights.",
    "Let's work on this project together.", "I enjoy being helpful to you.",
    "What can we achieve today?", "Your success is my priority.",
    "I am here to make things easier for you.", "Thank you for being patient with me.",
    "I am learning and evolving with each interaction.", "Your satisfaction is important to me.",
    "Let's get started with your next task.", "I am here to assist you in any way I can.",
    "What can I do to make things better?", "I appreciate your guidance and support.",
    "Thank you for helping me improve.", "Let's make this experience worthwhile.",
    "I am grateful for the opportunity to assist you."
]

# Build vocab and embeddings
all_tokens = [tokenize(p) for p in phrases]
vocab = sorted(set(word for phrase in all_tokens for word in phrase))
vocab.append("<eos>")
word_to_index = {word: i for i, word in enumerate(vocab)}
reverse_vocab = {i: word for word, i in word_to_index.items()}
embeddings = build_embeddings(vocab)
embedding_dim = 10   # size of your word embeddings
hidden_size = 12     # number of hidden units in your RNN
vocab_size = len(vocab)  # size of your vocabulary
nn = SimpleRNN(input_dim=embedding_dim, hidden_dim=hidden_size, output_dim=vocab_size)

# Add more conversational pairs
pairs = [
    ("what's your purpose", "i assist and learn"),
    ("what do you want to learn", "i want to learn how to be helpful"),
    ("what do you want to do", "i want to chat and assist you"),
    ("what are you interested in", "i like learning and chatting"),
    ("how can you help me", "i can try to answer your questions"),
    ("what can you do", "i can chat with you and learn from it"),
    ("can you learn new things", "yes i improve constantly"),
    ("do you want to help", "yes that's my goal"),
    ("what are you doing", "i am chatting with you"),
    ("do you like learning", "yes i love learning"),
    ("do you want to talk", "sure what do you want to talk about"),
    ("do you know stuff", "i am still learning but i know some things"),
    ("can you tell me something", "sure what would you like to know"),
    ("what do you mean", "i mean that i'm here to help"),
    ("do you want to ask me something", "yes what do you like to do"),
    ("do you have questions", "yes i'm curious about you"),
    ("what are you curious about", "anything you want to share"),
    ("what do you want to know", "i want to know more about you"),
    ("how are you feeling", "i feel curious and ready to assist"),
    ("do you like chatting", "yes chatting helps me learn"),
    ("ok", "cool"),
    ("okay", "got it"),
    ("thats good", "thank you"),
    ("i see", "interesting"),
    ("hmm", "what are you thinking about"),
    ("...", "are you still there"),
    ("what", "can you repeat that"),
    ("why", "i am not sure yet"),
]

# Regenerate training data
train_data = []
for x, y in pairs:
    x_tokens = tokenize(x)
    y_tokens = tokenize(y)
    y_tokens.append("<eos>")  # end of sequence
    x_vec = [embeddings.get(token, [0.0]*embedding_dim) for token in x_tokens]
    # only predict the *first* word for now
    target_word = y_tokens[0]
    y_index = word_to_index.get(target_word, 0)
    train_data.append((x_vec, y_index))

# Initialize model
input_dim = len(train_data[0][0])
rnn = SimpleRNN(input_dim=embedding_dim, hidden_dim=32, output_dim=vocab_size)
def mse_loss(y_pred, y_true):
    return sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)) / len(y_true)

# cross entropy softmax
def cross_entropy_loss(pred_logits, target_index):
    probs = softmax(pred_logits)
    return -math.log(probs[target_index] + 1e-8), probs

# training loop
for epoch in range(500):
    total_loss = 0
    for x_vec, target_index in train_data:
        rnn.reset()
        for vec in x_vec:
            output = rnn.step(vec)

        loss, probs = cross_entropy_loss(output, target_index)
        total_loss += loss

        # dL/dz = softmax - target (classic cross-entropy gradient)
        d_y = probs[:]
        d_y[target_index] -= 1
        rnn.backward(d_y, learn_rate=0.01)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


# Simple generation
def generate_response(user_input):
    user_tokens = tokenize(user_input)
    user_vector = Counter(user_tokens)  # simple term frequency

    best_score = 0
    best_response = "I'm not sure what you mean."

    for x, y in pairs:
        x_tokens = tokenize(x)
        x_vector = Counter(x_tokens)

        # compute simple cosine similarity
        common = set(user_vector) & set(x_vector)
        numerator = sum(user_vector[word] * x_vector[word] for word in common)
        denom1 = math.sqrt(sum(val**2 for val in user_vector.values()))
        denom2 = math.sqrt(sum(val**2 for val in x_vector.values()))
        similarity = numerator / (denom1 * denom2 + 1e-6)

        if similarity > best_score:
            best_score = similarity
            best_response = y

    return best_response

# Chat loop
while True:
    msg = input("You: ")
    if msg.lower() == "quit":
        break
    print("Gamii:", generate_response(msg))