from Code.bag_of_words import BagOfWords


class Parser:
    """ Parses the text into unigram/bigram words objects """
    # TODO: add stopwatch removal
    def __init__(self, fpath):
        self.path = fpath
        self.unigram_bag = BagOfWords()
        self.bigram_bag = BagOfWords()

        self.parse()

        self.sorted_unigram = self.unigram_bag.ordered()
        self.sorted_bigram = self.bigram_bag.ordered()

    def parse_unigram(self, line):
        """ add unigram words to unigram bag """
        for word in line.split(" "):
            self.unigram_bag.add(word)

    def parse_bigram(self, line):
        """ add bigram words to bigram bag """
        words = line.split(" ")
        for index in range(len(words) - 1):
            word = "{} {}".format(words[index], words[index + 1])
            self.bigram_bag.add(word)

    def parse(self):
        """ parse text to lines then into words """
        with open(self.path) as file:
            for line in file.readlines():
                clean_line = line.lower().strip()
                self.parse_unigram(clean_line)
                self.parse_bigram(clean_line)
