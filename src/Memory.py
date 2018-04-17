import numpy as np
import sys

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


'''
    episodes are lists of tuples (state, feat, action, reward, state', done) 
    where a state is a list of pixels, and feat is the game features
'''
class Replay_Buffer():
    e = 0.01  # epsilon ensures no transition has zero priority
    a = 0.6  # alpha determines the prioritization level. a = 0 is a uniform random sampling, a = 1 is priority only

    self.tree = None

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        
    def _get_priority(self, error):
        return (error + self.e) ** self.a

    # accepts a numpy array of 6-tuples
    def add(self, error, sample):
        #set the size of the tree on first addition, add() must be first operation
        if self.tree == None:
            ep_size = sys.getsizeof(sample) #in bytes
            capacity = self.max_bytes / ep_size
            self.tree = SumTree(capacity)

        p = self._get_priority(error)
        self.tree.add(p, sample)

    # returns a list of indexes, and a list of episodes. ([int], [[(s, f, a, r, sp, d)]])
    def sample(self, batch_size, trace_length):
        batch = list()
        indexes = list()
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            # get a full episode
            s = np.random.uniform(a, b)
            (idx, _, data) = self.tree.get(s)
            # choose a random segment from the episode.
            point = np.random.uniform(0, len(data) + 1 - trace_length)

            batch.append(data[point: point+trace_length])
            indexes.append(idx)
        
        batch = np.reshape(batch, [batch_size * trace_length, 6])
    
        return (indexes, batch)

    # after training on experience, need new error rate for it
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
