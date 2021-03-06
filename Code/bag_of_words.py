from Code.word import Word


class BagOfWords:
    """ Custom bag of words for naive bayes algorithm """
    def __init__(self):
        self.__bag = {}
        self.total = 0
        self.weights = 0

    def __getitem__(self, key):
        return self.__bag.get(key, Word(key, count=0))

    def __contains__(self, item):
        return item in self.__bag

    def __iter__(self):
        return iter(self.__bag)

    def __len__(self):
        return len(self.__bag)

    def add(self, word):
        """ add a new word to bag or update existing one, if the word is unique also increment the document counter """
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

        # update the total
        self.sum()

    def calculate_weights(self, number_of_documents=1):
        """
        document_count: number of training lines/documents
        update weights for every word in the bag
        """
        mx = max(self.__bag.values(), key=lambda x: x.count).count
        for word in self.__bag.values():
            word.calculate_tf_idf(mx, number_of_documents)
        self.sum()

    def size(self):
        """ get size of bag """
        return len(self.__bag)

    def from_dict(self, bag):
        """ directly assign a dict for bag, return the bag """
        self.__bag = bag
        self.sum()
        return self

    def __exclusive_filter(self, iterable):
        """ filtering based on excluding list of elements """
        new = self.__bag
        for item in iterable:
            if item in new:
                new.pop(item)
        return new

    def __inclusive_filter(self, iterable):
        """ filtering based on including list of elements """
        new = {}
        for item in self:
            if item in iterable:
                new[item] = self[item]
        return new

    def filter(self, iterable, method="exclusive"):
        """ filter bag using a list, return a filtered bag """
        out = BagOfWords()
        if method == "exclusive":
            return out.from_dict(self.__exclusive_filter(iterable))
        elif method == "inclusive":
            return out.from_dict(self.__inclusive_filter(iterable))
        else:
            print("Wrong method type")

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
            return list(set(self.__bag) | set(bag.bag()))
        elif method == "intersect":
            """ return a list of keys that exists in both bags """
            return list(set(self.__bag) & set(bag.bag()))
        elif method == "subtract":
            return list(set(self.__bag) - set(bag.bag()))
        else:
            print("Invalid method")
            return None

    def get(self, key):
        """ get a key from bag """
        return self.__bag.get(key, Word(key, count=0))

    def bag(self):
        """ get whole dictionary """
        return self.__bag

    def ordered(self, key=lambda x: x.count):
        """ return ordered list of keys """
        return sorted(list(self.__bag.values()), key=key, reverse=True)

    def values(self):
        """ return only values of dict """
        return list(self.__bag.values())

    def keys(self):
        """ return only the keys """
        return list(self.__bag.keys())

    def sum(self):
        """ update and return the sum of all elements """
        self.total = sum([word.count for word in self.__bag.values()])
        self.weights = sum([word.weight for word in self.__bag.values()])
        return self.total
