from Code.bag_of_words import BagOfWords


class Data:
    """ Holds unigram and bigram training/test data from files """
    def __init__(self, fpath=None, line_prepend="", line_append=""):
        self.unigram_bag = BagOfWords()
        self.bigram_bag = BagOfWords()

        self.line_prepend = line_prepend
        self.line_append = line_append

        self.file_path = fpath
        if fpath is not None:
            self.parse()

        self.sorted_unigram = []
        self.sorted_bigram = []

    def filter(self, iterable, method="exclusive"):
        """ filter the unigram bag """
        self.unigram_bag.filter(iterable, method=method)
        self.update()

    def update(self):
        """ update the sorted key lists """
        self.sorted_unigram = self.unigram_bag.ordered()
        self.sorted_bigram = self.bigram_bag.ordered()

    def parse_unigram(self, line):
        """ add unigram words from a line to unigram bag """
        for word in line.split(" "):
            self.unigram_bag.add(word)

    def parse_bigram(self, line):
        """ add bigram words from a line to bigram bag """
        words = line.split(" ")
        for index in range(len(words) - 1):
            word = "{} {}".format(words[index], words[index + 1])
            self.bigram_bag.add(word)

    def parse(self):
        """ add a whole plaintext into bag """
        with open(self.file_path) as f:
            for line in f.readlines():
                clean_line = "{}{}{}".format(self.line_prepend, line.lower().strip(), self.line_append)
                self.parse_unigram(clean_line)
                self.parse_bigram(clean_line)
        self.update()
