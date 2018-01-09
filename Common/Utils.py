import sys

def print_flush(s):
    print(s)
    sys.stdout.flush()
    
class IdAssigner:
    def __init__(self):
        self.forward = dict()
        self.reverse = dict()
        self.next_id = 0
    def get_id(self, x):
        if x not in self.forward:
            self.forward[x] = self.next_id
            self.reverse[self.next_id] = x
            self.next_id += 1
        return self.forward[x]
    def get_reverse_id(self, id_):
        return self.reverse[id_]
    def get_next_id(self):
        return self.next_id


