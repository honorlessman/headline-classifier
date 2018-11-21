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

    def accuracy(self):
        """ calculate accuracy """
        # for unigram
        self.results = {False: [line for line in self.test if line.is_fake != line.category],
                        True: [line for line in self.test if line.is_fake == line.category]}
        self.accuracy_score = len(list(self.results[True])) / (len(self.results[False]) + len(self.results[True]))
        self.accuracy_score *= 100

        # for bigram
        self.bi_results = {False: [line for line in self.test if line.bi_is_fake != line.category],
                           True: [line for line in self.test if line.bi_is_fake == line.category]}
        self.bi_accuracy_score = len(list(self.bi_results[True])) / (len(self.bi_results[False])
                                                                     + len(self.bi_results[True]))
        self.bi_accuracy_score *= 100

    def calculate_unigram(self, line, tf_idf=False):
        """ calculate real and fake score for given line and write scores to the line object itself """
        # summation of all the probability of words (with log10)
        fake_sums = 0.0
        real_sums = 0.0
        for word in line.unigram_bag.values():
            # for w being the word and c being class(real)
            # uni_fake_word_in_class: count(w,c) c = fake
            # uni_fake_total_in_class: count(c) c = fake
            if tf_idf:
                uni_fake_word_in_class = self.fake_parser.unigram_bag[word.word].weight
                uni_fake_total_in_class = self.fake_parser.unigram_bag.weights

                uni_real_word_in_class = self.real_parser.unigram_bag[word.word].weight
                uni_real_total_in_class = self.real_parser.unigram_bag.weights

            else:
                uni_fake_word_in_class = self.fake_parser.unigram_bag[word.word].count
                uni_fake_total_in_class = self.fake_parser.unigram_bag.total

                # for w being the word and c being class(real)
                # uni_fake_word_in_class: count(w,c) c = real
                # uni_fake_total_in_class: count(c) c = real
                uni_real_word_in_class = self.real_parser.unigram_bag[word.word].count
                uni_real_total_in_class = self.real_parser.unigram_bag.total

            # calculation is as follows:
            # count(w) * log_10
            # (
            # ( count(w,c) + 1   )
            #  -------------------
            # (count(c) + |v| + 1)
            # )
            fake_score = word.count * log10((uni_fake_word_in_class + 1) /
                                            (uni_fake_total_in_class + self.unigram_total + 1))

            # assign score to word and add it to the sum
            self.fake_parser.unigram_bag[word.word].score = fake_score
            fake_sums += fake_score

            real_score = word.count * log10((uni_real_word_in_class + 1) /
                                            (uni_real_total_in_class + self.unigram_total + 1))

            # assign score to word and add it to the sum
            self.real_parser.unigram_bag[word.word].score = real_score
            real_sums += real_score

        # assign the scores and predict
        line.fake_score = fake_sums
        line.real_score = real_sums
        line.predict()

    def calculate_bigram(self, line, tf_idf=False):
        """ calculate real and fake score for given line in bigram and write scores to the line object itself """
        # summation of all the probability of words (with log10)
        fake_sums = 0.0
        real_sums = 0.0
        for word in line.bigram_bag.values():
            # for w being the word and c being class(real)
            # bi_fake_word_in_class: count(w,c) c = fake
            # bi_fake_total_in_class: count(c) c = fake
            if tf_idf:
                bi_fake_word_in_class = self.fake_parser.bigram_bag[word.word].weight
                bi_fake_total_in_class = self.fake_parser.bigram_bag.weights

                bi_real_word_in_class = self.real_parser.bigram_bag[word.word].weight
                bi_real_total_in_class = self.real_parser.bigram_bag.weights
            else:
                bi_fake_word_in_class = self.fake_parser.bigram_bag[word.word].count
                bi_fake_total_in_class = self.fake_parser.bigram_bag.total

                # for w being the word and c being class(real)
                # bi_fake_word_in_class: count(w,c) c = real
                # bi_fake_total_in_class: count(c) c = real
                bi_real_word_in_class = self.real_parser.bigram_bag[word.word].count
                bi_real_total_in_class = self.real_parser.bigram_bag.total

            # calculation is as follows:
            # count(w) * log_10
            # (
            # ( count(w,c) + 1   )
            #  -------------------
            # (count(c) + |v| + 1)
            # )
            fake_score = word.count * log10((bi_fake_word_in_class + 1) /
                                            (bi_fake_total_in_class + self.bigram_total + 1))
            # assign score to word and add it to the sum
            fake_sums += fake_score
            self.fake_parser.bigram_bag[word.word].score = fake_score

            real_score = word.count * log10((bi_real_word_in_class + 1) /
                                            (bi_real_total_in_class + self.bigram_total + 1))
            # assign score to word and add it to the sum
            real_sums += real_score
            self.real_parser.bigram_bag[word.word].score = real_score

        # assign the scores and predict
        line.bi_fake_score = fake_sums
        line.bi_real_score = real_sums
        line.predict()

    def fit(self, line, tf_idf=False):
        self.test = line
        for line in self.test:
            # calculate bayes prediction for lines
            self.calculate_unigram(line, tf_idf=tf_idf)
            self.calculate_bigram(line, tf_idf=tf_idf)

        self.accuracy()
