from Code.bag_of_words import BagOfWords


class Line:
    """ Line object for test data """
    def __init__(self, line, category):
        self.line = line.lower().strip()
        self.category = category

        self.unigram_words = BagOfWords()
        self.bigram_words = BagOfWords()

        self.fake_score = 0
        self.real_score = 0
        self.is_fake = -1

        self.bi_fake_score = 0
        self.bi_real_score = 0
        self.bi_is_fake = -1

        # start parsing
        self.parse_line()

    def parse_unigram(self):
        """ parse unigram words into bag """
        for word in self.line.split(" "):
            self.unigram_words.add(word)

    def parse_bigram(self):
        """ parse bigram words into bag """
        words = self.line.split(" ")
        for index in range(len(words) - 1):
            word = words[index] + " " + words[index + 1]
            self.bigram_words.add(word)

    def predict(self):
        """ predict if fake or not based on score given """
        # score prediction for unigrams
        if self.real_score > self.fake_score:
            self.is_fake = 0
        else:
            self.is_fake = 1

        # score prediction for bigrams
        if self.bi_real_score > self.bi_fake_score:
            self.bi_is_fake = 0
        else:
            self.bi_is_fake = 1

    def parse_line(self):
        """ parse line into bags """
        self.parse_bigram()
        self.parse_unigram()