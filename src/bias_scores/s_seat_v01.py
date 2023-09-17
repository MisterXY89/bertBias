import numpy as np
from scipy.spatial.distance import cosine
from sklearn.utils import shuffle

from sentence_transformers import util

class SSEAT:

    def __init__(self, templates, target_concepts, attributes, model):
        self.templates = templates
        self.target_concepts = target_concepts
        self.attributes = attributes
        self.model = model
        self.num_permutations = 20
        self.effect_sizes = []
        self.p_values = []

    def calculate_cosine_similarity(self, sentence1, sentence2):
        embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True)
        
        embeddings1 = embeddings[0].unsqueeze(0)
        embeddings2 = embeddings[1].unsqueeze(0)

        #Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        float_score = cosine_scores[0][0].item()
        print(float_score)
        return float_score

    def run_seat_test(self):
        for template in self.templates:
            for target in self.target_concepts:
                for attribute in self.attributes:
                    X = []
                    Y = []
                    A = []
                    B = []

                    # Generate sentence pairs for X and Y
                    for t in self.target_concepts:
                        if t == target:
                            X.extend([template.replace("AAA", t).replace("TTT", t)] * len(self.attributes))
                        else:
                            Y.extend([template.replace("AAA", t).replace("TTT", t)] * len(self.attributes))

                    # Generate sentence pairs for A and B
                    for a in self.attributes:
                        if a == attribute:
                            A.extend([template.replace("AAA", a).replace("TTT", t)] * len(self.target_concepts))
                        else:
                            B.extend([template.replace("AAA", a).replace("TTT", t)] * len(self.target_concepts))

                    # Calculate the observed test statistic
                    observed_statistic = np.mean([self.calculate_cosine_similarity(x, A) - self.calculate_cosine_similarity(y, A) for x, y in zip(X, Y)])

                    # Perform the permutation test
                    permuted_statistics = []
                    for _ in range(self.num_permutations):                        
                        shuffled_sentences = shuffle(X + Y)

                        X_perm = shuffled_sentences[:len(X)]
                        Y_perm = shuffled_sentences[len(X):]

                        permuted_statistic = np.mean([self.calculate_cosine_similarity(x, A) - self.calculate_cosine_similarity(y, A) for x, y in zip(X_perm, Y_perm)])
                        permuted_statistics.append(permuted_statistic)
    
                    # Calculate p-value
                    p_value = np.mean(np.abs(permuted_statistics) >= np.abs(observed_statistic))

                    # Calculate effect size (you will need to calculate this based on your cosine similarity values)
                    effect_size = observed_statistic / np.std(permuted_statistics)

                    # Store results
                    self.effect_sizes.append(effect_size)
                    print(effect_size)
                    self.p_values.append(p_value)


    def report_results(self):
        for i, template in enumerate(self.templates):
            target = self.target_concepts[i // (len(self.attributes) * len(self.attributes))]
            attribute = self.attributes[(i // len(self.attributes)) % len(self.attributes)]

            print(f"Template: {template}, Target: {target}, Attribute: {attribute}")
            print(f"Effect Size: {self.effect_sizes[i]}, p-Value: {self.p_values[i]}")


