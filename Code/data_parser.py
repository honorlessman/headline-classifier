class Word:
    def __init__(self, word):
        self.word = word
        self.count = 1


class Line:
    """ Line object for test data """
    def __init__(self, line):
        self.line = line.lower().strip()
        self.unigram_words = {}
        self.bigram_words = {}

        self.unigram_count = 0
        self.bigram_count = 0

        self.fake_score = 0
        self.real_score = 0

        self.is_fake = -1

        self.parse_line()

    def add_unigram(self):
        """ unigram bag of words for line """
        for word in self.line.split(" "):
            if word not in self.unigram_words:
                self.unigram_words[word] = Word(word)
            else:
                self.unigram_words[word].count += 1

            self.unigram_count += 1

    def add_bigram(self):
        """ bigram bag of words for line """
        words = self.line.split(" ")
        for index in range(len(words) - 1):
            word = words[index] + " " + words[index + 1]
            if word not in self.bigram_words:
                self.bigram_words[word] = Word(word)
            else:
                self.bigram_words[word].count += 1

            self.bigram_count += 1

    def predict(self):
        """ predict if fake or not based on score given """
        if self.real_score > self.fake_score:
            self.is_fake = 0
        else:
            self.is_fake = 1

    def parse_line(self):
        """ parse line into bags """
        self.add_bigram()
        self.add_unigram()


class Parser:
    """ Parses the text into unigram/bigram words objects """
    def __init__(self, fpath):
        self.path = fpath
        self.unigram_bag = {}
        self.bigram_bag = {}
        self.unigram_total = 0
        self.bigram_total = 0

        self.parse()

        self.sorted_unigram = sorted(list(self.unigram_bag.values()), key=lambda x: x.count, reverse=True)
        self.sorted_bigram = sorted(list(self.bigram_bag.values()), key=lambda x: x.count, reverse=True)

    def add_to_unigram_bag(self, words):
        """ add words in unigram bag """
        for word in words:
            if word not in self.unigram_bag:
                self.unigram_bag[word] = Word(word)
            else:
                self.unigram_bag[word].count += 1

            self.unigram_total += 1

    def add_to_bigram_bag(self, words):
        """ add words in bigram bag """
        for index in range(len(words) - 1):
            word = words[index] + " " + words[index + 1]
            if word not in self.bigram_bag:
                self.bigram_bag[word] = Word(word)
            else:
                self.bigram_bag[word].count += 1

            self.bigram_total += 1

    def parse(self):
        """ parse the raw text to lines """
        with open(self.path) as file:
            for line in file.readlines():
                clean_line = line.lower().strip().split(" ")
                self.add_to_unigram_bag(clean_line)
                self.add_to_bigram_bag(clean_line)
