# a test file for algorithms
import json

"""
id3 implements the iterative dichomotizer 3 algorithm and constructs an optimal
decision tree for classification of the data. Returns a tree in the form of XXX.
features is a dict mapping from feature names to their value spaces
samples is a list where each row contains a data sample
"""
class Node:
    def __init__(self, condition, value):
        self.condition = condition
        self.value = value 
        self.children = []

    def evaluate(self, X):
        return condition(X)

    def is_leaf(self):
        return len(self.children) == 0

class DecisionTree:
    
    def __init__(self, root):
        self.root = root 

    def predict(self, X):
        queue = [self.root]
        curr = 0

        while True:
            node = queue[curr]
            if node.is_leaf():
                return node.value
            elif node.evaluate(X):
                queue.extend(node.children)
            curr += 1

        return None


def NewDecisionTreeWithID3():
    tree = DecisionTree()

    return tree

def id3(features, samples):
    print samples

# build a mapping from value spaces of features to the count of associated outcomes
def build_ftr_map(X, y):
    values_to_outcomes = {}
    


def 



if __name__ == "__main__":
    with open('test_data.json', 'r') as datfile:
        data = json.loads(datfile.read())
        ftrs, samples = data.values()
        X, y = samples[:-1], samples[-1]

        # collect the value space for each feature by inspecting the
        # training samples
        ftr_values = {}
        for idx, ftr in enumerate(ftrs):
            ftr_values[ftr] = list(set([s[idx] for s in samples]))

        
        

        print ftr_values 
        print values_to_samples

        dtree = id3(ftrs, samples)