######------------------------A Detailed Description Of How The Approach Works-------------------------------######
We have used TransE Knowledge Graph Embedding model for fact checking in the knowledge graph.

Knowledge Graph Embedding represents the entities and relations of a knowledge graph with a high dimension vector. These vectors preserves the features of individual entities and relations. 

In TransE embedding model the vector representations follows the following rule -
Head embedding vector + Relation embedding vector ≈ Tail embedding vector

Based on this we can calculate a "score" for each triple.
score = euclidean_distance(Head embedding vector + Relation embedding vector - Tail embedding vector)
This equation is called the "Scoring function" of TransE model.
Our goal is to minimize the score for positive triple (true facts) and maximize it for negative triple (false facts). 

#----------Embedding procedure using machine learning-----------------#
For each positive triple (true fact) a random negative triple is sampled. This is called negative sampling. We have done it by replacing the head or tail or both with a random corrupt head or tail respectively.
For each negative triple (false fact) a random positive triple is sampled. This is called positive sampling. We have done it by sampling a triple that contains the particular (head and relation) or (relation and tail).

Now for these pair of positive and negative triples, a positive and negative score is calculated and the embeddings are updated using the loss function. We have used margin based ranking loss function.
Loss=max(0,score_positive−score_negative+margin)

--> The hyperparameters we used are mentioned below -
Embedding Dimension = 50
Margin = 1.0    
Epochs = 100 (trained model two times with 50 epochs each)
Learning Rate = 0.01

#######-------------------------Clear Instructions On How To Execute The Code----------------------------######
Steps:-
1) Install python. (We used python version 3.12.7)
2) Install the following packages in python.
   • numpy (pip install numpy)
   • pytorch (pip install torch torchvision torchaudio)
   • rdflib (pip install rdflib)
   • tqdm (pip install tqdm)
#---------------Below are further instructions to Test the model-------------------------#
3) Place the following files in one folder.
   • KnowledgeGraphProject.py  (this is the main script)
   • transe_model.pth (Unzip this file from transe_model.zip in the same folder) (this file contains the   trained model weights and other parameters)
   • fokg-sw-test-2024.nt (this file contains the test statements for fact checking)
4) Open command prompt in the folder and execute (python KnowledgeGraphProject.py)
5) A menu will appear in the console. Press 2 for "Test Model".
6) A result.ttl file will be generated that you can upload on GERBIL to check the ROC AUC Curve.

NOTE: If you want to train the model by yourself from start then the instructions for that are given below. However it might take several hours to train the model.
#---------------Below are instructions to Train the model--------------------------------#
Step 1) and 2) remains the same.
3) Place the following files in one folder.
   • KnowledgeGraphProject.py
   • transe_model.pth
   • fokg-sw-train-2024.nt (this file contains the labeled facts for training)
   • reference-kg.nt (this file contains the Knowledge graph RDF in n-triple format)
4) Open KnowledgeGraphProject.py and on line 280, enter the number of epochs you want to train (recommended 100 epochs). Save and close the file.
5) Open command prompt in the folder and execute (python KnowledgeGraphProject.py)
5) A menu will appear in the console. Press 1 for "Train Model".
6) Console will show training progress percentage and epochs remaining.
6) The transe_model.pth file will be updated with trained weights or a new transe_model.pth will be created if not provided in the folder.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Here we are providing the link to the GERBIL Experiment of the generated result.ttl file.
Link - https://gerbil-kbc.aksw.org/gerbil/experiment?id=202501150032
Thankyou.
