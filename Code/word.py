from math import log10


class Word:
    """ a unigram/bigram word with counter """
    def __init__(self, word, count=1):
        self.word = word
        self.count = count

        # The score calculated in bayes
        self.score = 0

        # TF IDF parameters
        self.weight = 0
        self.existing_document_count = 1

    def calculate_tf_idf(self, total_word_count_of_document, number_of_documents):
        """ calculate term frequency * inverse document frequency """
        tf = log10(1 + (self.count / total_word_count_of_document))
        idf = abs(log10(self.existing_document_count / number_of_documents))
        self.weight = tf * idf
