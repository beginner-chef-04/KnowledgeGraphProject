import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import rdflib
from rdflib import Graph
import os
import sys 
import random
import warnings

# Define the TransE model
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, head, relation, tail, mode='positive'):
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)

        if mode == 'positive':
            score = torch.norm(head_emb + relation_emb - tail_emb, p=2, dim=1)
        else:  # Negative samples
            score = torch.norm(head_emb + relation_emb - tail_emb, p=2, dim=1)
        return score

    def get_embeddings(self):
        return self.entity_embeddings.weight.detach().numpy(), self.relation_embeddings.weight.detach().numpy()

# Define the loss function for TransE
class MarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin)

    def forward(self, positive_scores, negative_scores):
        target = torch.tensor([-1], dtype=torch.float32)  # Negative ranking loss target
        return self.loss_fn(positive_scores, negative_scores, target)

# Prepare training data
def prepare_data(KGtriples, LFtriples):
    #----------Mapping Entities and Relations to index----------#
    entity_set = set()
    relation_set = set()
    for head, relation, tail in KGtriples:
        entity_set.add(head)
        entity_set.add(tail)
        relation_set.add(relation)

    entity_to_idx = {entity: idx for idx, entity in enumerate(entity_set)}
    relation_to_idx = {relation: idx for idx, relation in enumerate(relation_set)}
    #---------- Merge KnowledgeGraph data and LabeledFacts data together-----------------#
    KGdata = [(entity_to_idx[head], relation_to_idx[relation], entity_to_idx[tail], int(2)) for head, relation, tail in KGtriples]
    LFdata = [(entity_to_idx[head], relation_to_idx[relation], entity_to_idx[tail], labelValue) for head, relation, tail, labelValue in LFtriples]
    
    data = [*LFdata, *KGdata]
    random.shuffle(data)
    return data, entity_to_idx, relation_to_idx

# For Positive Sampling
def generate_positive_samples(negative_triple, num_entities, idx_triples):
    positive_samples = []
    head, relation, tail, labelValue = negative_triple

    # For correct head and relation (head, relation, *, int(2))
    correct_head_relation_triples = [t for t in idx_triples if t[0] == head and t[1] == relation and t[-1] == int(1)]
    positive_samples.extend(correct_head_relation_triples)

    # For correct relation and tail (*, relation, tail, int(2))
    correct_relation_tail_triples = [t for t in idx_triples if t[1] == relation and t[2] == tail and t[-1] == int(1)]
    positive_samples.extend(correct_relation_tail_triples)

    # For random positive triple
    if positive_samples == []:
        random_positive_triple = [t for t in idx_triples if t[-1] == int(1)]
        random.shuffle(random_positive_triple)
        positive_samples.append(random_positive_triple[0])
    
    return positive_samples

# For Negative Sampling
def generate_negative_samples(positive_triple, num_entities, idx_triples):
    negative_samples = []
    head, relation, tail, labelValue = positive_triple
    
    # For corrupt head
    while True:
        corrupt_head = torch.randint(0, num_entities, (1,)).item()
        corrupt_head_triple = (corrupt_head, relation, tail, int(1))
        if corrupt_head_triple not in idx_triples:
            negative_samples.append(corrupt_head_triple)
            break
    
    # For corrupt tail
    while True:
        corrupt_tail = torch.randint(0, num_entities, (1,)).item()
        corrupt_tail_triple = (head, relation, corrupt_tail, int(1))
        if corrupt_tail_triple not in idx_triples:
            negative_samples.append(corrupt_tail_triple)
            break
    
    return negative_samples

# Training the TransE model
def train_transe(model, optimizer, idx_triples, num_entities, num_relations, epochs):
    loss_fn = MarginRankingLoss(margin)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for head, relation, tail, labelValue in tqdm(idx_triples):
            if labelValue == 2 or labelValue == 1:
                negative_samples = generate_negative_samples((head, relation, tail, labelValue), num_entities, idx_triples)
                for corrupt_head, corrupt_relation, corrupt_tail, neg_samp_labelValue in negative_samples:
                    positive_score = model(torch.tensor([head]), torch.tensor([relation]), torch.tensor([tail]), mode='positive')                
                    negative_score = model(torch.tensor([corrupt_head]), torch.tensor([corrupt_relation]), torch.tensor([corrupt_tail]), mode='negative')
                    loss = loss_fn(positive_score, negative_score)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            else:
                positive_samples = generate_positive_samples((head, relation, tail, labelValue), num_entities, idx_triples)
                for correct_head, correct_relation, correct_tail, pos_samp_labelValue in positive_samples:
                    negative_score = model(torch.tensor([head]), torch.tensor([relation]), torch.tensor([tail]), mode='negative')                
                    positive_score = model(torch.tensor([correct_head]), torch.tensor([correct_relation]), torch.tensor([correct_tail]), mode='positive')
                    loss = loss_fn(positive_score, negative_score)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model

# Save the Trained model
def save_model(model, optimizer, data, entity_to_idx, relation_to_idx, learning_rate, TMfile_path):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "data": data,
        "entity_to_idx": entity_to_idx,
        "relation_to_idx": relation_to_idx,
        "embedding_dim": model.entity_embeddings.embedding_dim,
        "margin": model.margin,
        "learning_rate": learning_rate
    }
    torch.save(state, TMfile_path)
    print(f"Model saved to {TMfile_path}")

# Load the trained model (for testing or further training)
def load_model(TMfile_path):
    warnings.filterwarnings("ignore", category=FutureWarning)
    state = torch.load(TMfile_path)

    num_entities = len(state["entity_to_idx"])
    num_relations = len(state["relation_to_idx"])
    embedding_dim = state["embedding_dim"]
    margin = state["margin"]
    learning_rate = state["learning_rate"]
    model = TransE(num_entities, num_relations, embedding_dim, margin)
    model.load_state_dict(state["model_state_dict"])
    optimizer = Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    print(f"Model loaded from {TMfile_path}")

    return model, optimizer, state["data"], state["entity_to_idx"], state["relation_to_idx"]

# Maps the reification statements in Labeled Facts to corresponding triples
def statement_to_LFtriples(LFgraph):
    LFtriples = []
    stmt_to_triple = {}

    for statement in LFgraph.subjects(predicate=rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")):
        head = LFgraph.value(subject=statement, predicate=rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"))
        relation = LFgraph.value(subject=statement, predicate=rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"))
        tail = LFgraph.value(subject=statement, predicate=rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#object"))
        labelValue = int(float(LFgraph.value(subject=statement, predicate=rdflib.URIRef("http://swc2017.aksw.org/hasTruthValue"))))
        LFtriples.append((head, relation, tail, labelValue))
        stmt_to_triple[statement] = (head, relation, tail, labelValue)

    return LFtriples, stmt_to_triple

# Generates the scores for the test fact triples
def test_triples(TFtriples, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings):
    warnings.filterwarnings("ignore", category=UserWarning)
    TFtriples_to_scores = {}

    for triple in TFtriples:
        head, relation, tail = triple

        head_emb = entity_embeddings[entity_to_idx[head]]
        relation_emb = relation_embeddings[relation_to_idx[relation]]
        tail_emb = entity_embeddings[entity_to_idx[tail]]
        
        score = torch.norm(torch.tensor([head_emb], dtype=torch.float64) + torch.tensor([relation_emb], dtype=torch.float64) - torch.tensor([tail_emb], dtype=torch.float64), p=2, dim=1)

        TFtriples_to_scores[triple] = score.item()
    
    return TFtriples_to_scores

# Import and extract data from n-triple file to test
def import_test_data(TFfile_path):
    TFgraph = Graph()
    TFgraph.parse(TFfile_path, format='nt')  # You can also use 'turtle' or 'nt' for other formats

    TFtriples = []
    stmt_to_TFtriples = {}

    for statement in TFgraph.subjects(predicate=rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")):
        head = TFgraph.value(subject=statement, predicate=rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"))
        relation = TFgraph.value(subject=statement, predicate=rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"))
        tail = TFgraph.value(subject=statement, predicate=rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#object"))

        TFtriples.append((head, relation, tail))
        stmt_to_TFtriples[statement] = (head, relation, tail)

    return TFtriples, stmt_to_TFtriples

# Generate the result.ttl file for the test fact triples
def generate_result_file(stmt_to_TFtriples, TFtriples_to_scores, Resultfile_path):

    if os.path.exists(Resultfile_path):
        os.remove(Resultfile_path)

    stmt_to_scores = {}
    for stmt, TFtriples in stmt_to_TFtriples.items():
        score = TFtriples_to_scores[TFtriples]
        stmt_to_scores[stmt] = score
        output_string = f"""<{stmt}> <http://swc2017.aksw.org/hasTruthValue> "{score}"^^<http://www.w3.org/2001/XMLSchema#double>.\n"""
        open(Resultfile_path, "a").write(output_string)
    
    return

# Normalize the scores of test triples
def score_normalization(TFtriples_to_scores):
    score_list = []
    for TFtriples, score in TFtriples_to_scores.items():
        score_list.append(score)

    scores = np.array(score_list)

    min_score = scores.min()
    max_score = scores.max()
    normalized_scores = (scores - min_score) / (max_score - min_score)
    normalized_scores = 1 - normalized_scores
    normalized_scores = np.round(normalized_scores, decimals=1)

    for TFtriple, normalized_score in zip(TFtriples_to_scores.keys(), normalized_scores.tolist()):
        TFtriples_to_scores[TFtriple] = normalized_score
    
    return TFtriples_to_scores

# Program starts here
if __name__ == "__main__":
    #Current Directory Path
    current_directory = os.path.dirname(os.path.abspath(sys.argv[0])) 

    print("\n--- Model Menu ---")
    print("1. Train Model")
    print("2. Test Model")
    print("3. Exit")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == '1':    # To Train model
        # Model Hyperparameters
        embedding_dim = 50
        margin = 1.0    
        epochs = 50
        learning_rate = 0.01

        # Trained model file path
        TMfile_name="transe_model.pth"
        TMfile_path = os.path.join(current_directory,TMfile_name)

        # Check Trained model Exist
        if os.path.isfile(TMfile_path):
            print(f"Trained Model file path : {TMfile_path}")
            # Load model and data
            model, optimizer, data, entity_to_idx, relation_to_idx = load_model(TMfile_path)
        else:
            # Initialize the rdf graph
            KGgraph = Graph()
            LFgraph = Graph()

            # reference knowledge graph and labeled facts file path
            KGfile_name = 'reference-kg.nt' #Enter the name of the Knowledge Graph
            KGfile_path = os.path.join(current_directory,KGfile_name)
            LFfile_name = 'fokg-sw-train-2024.nt' #Enter the name of the Labeled Facts training data
            LGfile_path = os.path.join(current_directory,LFfile_name)

            # Parse trough RDF Graph
            KGgraph.parse(KGfile_path, format='nt')  # You can also use 'turtle' or 'nt' for other formats
            LFgraph.parse(LGfile_path, format='nt')

            # Convert the graph to a list of triples
            KGtriples = list(KGgraph)
            LFtriples, stmt_to_triple = statement_to_LFtriples(LFgraph)        

            # Prepare training data
            data, entity_to_idx, relation_to_idx = prepare_data(KGtriples, LFtriples)
            num_entities=len(entity_to_idx)
            num_relations=len(relation_to_idx)

            # Instantiate the model
            model = TransE(num_entities, num_relations, embedding_dim, margin)
            optimizer = Adam(model.parameters(), lr=learning_rate)

        # Train the model
        trained_model = train_transe(model,
            optimizer=optimizer,
            idx_triples=data,
            num_entities=len(entity_to_idx),
            num_relations=len(relation_to_idx),
            epochs=epochs
        )

        # Save the trained model
        save_model(trained_model, optimizer, data, entity_to_idx, relation_to_idx, learning_rate, TMfile_path)

    elif choice == '2':    # To Test model
        # Import Trained model
        TMfile_name="transe_model.pth"
        TMfile_path = os.path.join(current_directory,TMfile_name)
        model, optimizer, data, entity_to_idx, relation_to_idx = load_model(TMfile_path)

        # Retrieve embeddings
        entity_embeddings, relation_embeddings = model.get_embeddings()

        # n-triple test file path
        TFfile_name = 'fokg-sw-test-2024.nt'
        TFfile_path = os.path.join(current_directory,TFfile_name)

        # Import test data from n-triple test file
        TFtriples, stmt_to_TFtriples = import_test_data(TFfile_path)

        # Generate and Normalize the scores
        TFtriples_to_scores = test_triples(TFtriples, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings)
        TFtriples_to_scores = score_normalization(TFtriples_to_scores)

        # Output result file path
        Resultfile_name = 'result.ttl'
        Resultfile_path = os.path.join(current_directory,Resultfile_name)

        # Generate the output result.ttl file
        generate_result_file(stmt_to_TFtriples, TFtriples_to_scores, Resultfile_path)

    elif choice == '3':
        print("Exiting the program. Goodbye!")

    else:
        print("Invalid choice. Please enter 1, 2, or 3.")