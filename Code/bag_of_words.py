from Code.word import Word


class BagOfWords:
    def __init__(self):
        self.__bag = {}
        self.total = 0

    def __getitem__(self, key):
        return self.__bag.get(key, Word(key, count=0))

    def add(self, word):
        """ add a new word to bag or update existing one """
        if word not in self.__bag:
            self.__bag[word] = Word(word)
        else:
            self.__bag[word].count += 1

        self.total += 1

    def remove(self, word):
        """ remove word from bag if it exists """
        if word not in self.__bag:
            print("No such word exists in bag")
        else:
            self.__bag.pop(word)

    def size(self):
        """ get size of bag """
        return len(self.__bag)

    def from_dict(self, bag):
        """ directly assign a dict for bag, return the bag """
        self.__bag = bag
        return self

    def merge(self, bag, method="combine"):
        """ merge bags """
        if method == "combine":
            """ merge two bags together combining similar keys return a new bag """
            merged_dict = {word: Word(word, count=self.get(word).count + bag.get(word).count)
                           for word in set(self.__bag) | set(bag.bag())}
            out = BagOfWords().from_dict(merged_dict)
            return out
        elif method == "keys":
            """ if you only need keys return set of keys instead, usually faster """
            return set(self.__bag) | set(bag.bag())

    def get(self, key):
        """ get a key from bag """
        return self.__bag.get(key, Word(key, count=0))

    def bag(self):
        """ get whole dictionary """
        return self.__bag

    def ordered(self):
        """ return ordered list of keys """
        return sorted(list(self.__bag.values()), key=lambda x: x.count, reverse=True)

    def values(self):
        """ return only values of dict """
        return list(self.__bag.values())

    def keys(self):
        """ return only the keys """
        return list(self.__bag.keys())
