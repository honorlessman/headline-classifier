class Word:
    """ a unigram/bigram word with counter """
    def __init__(self, word, count=1):
        self.word = word
        self.count = count

    def append(self, string):
        """ append some string to behind of the word """
        self.word = "{} {}".format(self.word, string)

    def prepend(self, string):
        """ prepend some string in front of the word """
        self.word = "{} {}".format(string, self.word)
