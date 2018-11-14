class NaiveBayes:
    """ Naive bayes model with objects as data """
    def __init__(self, parser):
        self.fake_parser = parser[0]
        self.real_parser = parser[1]
        self.test = []

        self.unigram_total = self.fake_parser.unigram_total + self.real_parser.unigram_total
        self.bigram_total = self.fake_parser.bigram_total + self.real_parser.bigram_total

        self.fake_possibility = self.fake_parser.unigram_total / self.unigram_total
        self.fake_possibility_bigram = self.fake_parser.bigram_total / self.bigram_total

        self.real_possibility = self.real_parser.unigram_total / self.unigram_total
        self.real_possibility_bigram = self.real_parser.bigram_total / self.bigram_total

        self.unigram_score = 0

    def calculate_fake(self, line):
        """ calculate the possibility of line to be fake or real """
        # multiplication of the possibility of words in bag of words
        possibilities_of_bow = 1

        # multiplication of the possibility of words from bag of words to be fake/real
        possibilities_of_bow_fake = 1
        possibilities_of_bow_real = 1

        for word in line.unigram_words.values():
            # counts of words from train bag of words
            fake_count = self.fake_parser.unigram_bag[word.word].count if word.word in self.fake_parser.unigram_bag else 1
            real_count = self.real_parser.unigram_bag[word.word].count if word.word in self.real_parser.unigram_bag else 1

            possibilities_of_bow *= (word.count * (fake_count + real_count) / self.unigram_total)
            possibilities_of_bow_fake *= (word.count * fake_count / self.unigram_total)
            possibilities_of_bow_real *= (word.count * real_count / self.unigram_total)

        line.fake_score = possibilities_of_bow_fake * self.fake_possibility / possibilities_of_bow
        line.real_score = possibilities_of_bow_real * self.real_possibility / possibilities_of_bow
        line.predict()

    def fit(self, line):
        self.test = line

        for line in self.test:
            self.calculate_fake(line)
