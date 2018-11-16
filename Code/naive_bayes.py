from math import log10


class NaiveBayes:
    """ Naive bayes model for headlines.Uses objects as data """
    def __init__(self, parser):
        self.fake_parser = parser[0]
        self.real_parser = parser[1]
        self.test = []

        self.results = {}
        self.accuracy_score = 0.0

        self.bi_results = {}
        self.bi_accuracy_score = 0.0

        self.unigram_total = len(self.real_parser.unigram_bag.merge(self.fake_parser.unigram_bag, method="keys"))
        self.bigram_total = len(self.real_parser.bigram_bag.merge(self.fake_parser.bigram_bag, method="keys"))

        self.fake_possibility = self.fake_parser.unigram_bag.size() / self.unigram_total
        self.fake_possibility_bigram = self.fake_parser.bigram_bag.size() / self.bigram_total

        self.real_possibility = self.real_parser.unigram_bag.size() / self.unigram_total
        self.real_possibility_bigram = self.real_parser.bigram_bag.size() / self.bigram_total

        self.unigram_score = 0

    def accuracy(self):
        """ calculate accuracy """
        self.results = {False: [line for line in self.test if line.is_fake != line.category],
                        True: [line for line in self.test if line.is_fake == line.category]}
        self.accuracy_score = len(list(self.results[True])) / (len(self.results[False]) + len(self.results[True]))

        self.bi_results = {False: [line for line in self.test if line.bi_is_fake != line.category],
                           True: [line for line in self.test if line.bi_is_fake == line.category]}
        self.bi_accuracy_score = len(list(self.bi_results[True])) / (len(self.bi_results[False]) + len(self.bi_results[True]))

    def old_calculate_fake(self, line):
        """ calculate the possibility of line to be fake or real """
        # multiplication of the possibility of words in bag of words
        possibilities_of_bow = 1

        # multiplication of the possibility of words from bag of words to be fake/real
        possibilities_of_bow_fake = 1
        possibilities_of_bow_real = 1

        for word in line.unigram_words.values():
            # counts of words from train bag of words
            fake_count = self.fake_parser.unigram_bag[word.word].count
            real_count = self.real_parser.unigram_bag[word.word].count

            possibilities_of_bow *= (word.count * (fake_count + real_count) / self.unigram_total)
            possibilities_of_bow_fake *= (word.count * fake_count * self.fake_possibility / self.unigram_total)
            possibilities_of_bow_real *= (word.count * real_count * self.real_possibility / self.unigram_total)

        line.fake_score = possibilities_of_bow_fake / possibilities_of_bow
        line.real_score = possibilities_of_bow_real / possibilities_of_bow
        line.predict()

    def calculate_unigram(self, line):
        """ calculate real and fake score for given line and write scores to the line object itself """
        # summation of all the probability of words (with log10)
        fake_sums = 0.0
        real_sums = 0.0
        for word in line.unigram_words.values():
            # for w being the word and c being class(real)
            # uni_fake_word_in_class: count(w,c) c = fake
            # uni_fake_total_in_class: count(c) c = fake
            uni_fake_word_in_class = self.fake_parser.unigram_bag[word.word].count
            uni_fake_total_in_class = self.fake_parser.unigram_bag.total

            # for w being the word and c being class(real)
            # uni_fake_word_in_class: count(w,c) c = real
            # uni_fake_total_in_class: count(c) c = real
            uni_real_word_in_class = self.real_parser.unigram_bag[word.word].count
            uni_real_total_in_class = self.real_parser.unigram_bag.total

            # calculation is as follows:
            # log_10
            # (
            # ( count(w,c) + 1   )
            #  -------------------
            # (count(c) + |v| + 1)
            # )
            fake_sums += log10((uni_fake_word_in_class + 1) / (uni_fake_total_in_class + self.unigram_total + 1))
            real_sums += log10((uni_real_word_in_class + 1) / (uni_real_total_in_class + self.unigram_total + 1))

        # assign the scores and predict
        line.fake_score = fake_sums
        line.real_score = real_sums
        line.predict()

    def calculate_bigram(self, line):
        """ calculate real and fake score for given line in bigram and write scores to the line object itself """
        # summation of all the probability of words (with log10)
        fake_sums = 0.0
        real_sums = 0.0
        for word in line.bigram_words.values():
            # for w being the word and c being class(real)
            # bi_fake_word_in_class: count(w,c) c = fake
            # bi_fake_total_in_class: count(c) c = fake
            bi_fake_word_in_class = self.fake_parser.bigram_bag[word.word].count
            bi_fake_total_in_class = self.fake_parser.bigram_bag.total

            # for w being the word and c being class(real)
            # bi_fake_word_in_class: count(w,c) c = real
            # bi_fake_total_in_class: count(c) c = real
            bi_real_word_in_class = self.real_parser.bigram_bag[word.word].count
            bi_real_total_in_class = self.real_parser.bigram_bag.total

            # calculation is as follows:
            # log_10
            # (
            # ( count(w,c) + 1   )
            #  -------------------
            # (count(c) + |v| + 1)
            # )
            fake_sums += log10((bi_fake_word_in_class + 1) / (bi_fake_total_in_class + self.bigram_total + 1))
            real_sums += log10((bi_real_word_in_class + 1) / (bi_real_total_in_class + self.bigram_total + 1))

        # assign the scores and predict
        line.bi_fake_score = fake_sums
        line.bi_real_score = real_sums
        line.predict()

    def fit(self, line):
        self.test = line
        for line in self.test:
            # calculate bayes prediction for lines
            self.calculate_unigram(line)
            self.calculate_bigram(line)

        self.accuracy()
