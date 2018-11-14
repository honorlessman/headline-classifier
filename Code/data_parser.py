class Parser:
    """ Parses the text into unigram words objects """
    def __init__(self, fpath):
        self.path = fpath
        self.unigram_bag = {}
        self.bigram_bag = {}

        self.parse()
        self.sorted_bag = sorted(list(self.unigram_bag.values()), key=lambda x: x.count, reverse=True)

    def add_to_unigram_bag(self, words):
        """ add words in both bigram and unigram """
        for word in words:
            if word not in self.unigram_bag:
                self.unigram_bag[word] = Word(word)
            else:
                self.unigram_bag[word].count += 1

    def add_to_bigram_bag(self, words):
        pass

    def parse(self):
        """ parse the raw text to lines """
        with open(self.path) as file:
            for line in file.readlines():
                clean_line = line.lower().strip().split(" ")
                self.add_to_unigram_bag(clean_line)
                self.add_to_bigram_bag(clean_line)


class Word:
    def __init__(self, word):
        self.word = word
        self.count = 1
