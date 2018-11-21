from Code.bag_of_words import BagOfWords


class Data:
    """ Holds unigram and bigram training/test data from files """

    def __init__(self, fpath=None, line_prepend="", line_append=""):
        self.unigram_bag = BagOfWords()
        self.bigram_bag = BagOfWords()

        self.line_count = 0

        self.line_prepend = line_prepend
        self.line_append = line_append

        self.result = {
            "unigram top 10": [],
            "bigram top 10": [],
            "stopword top 10": [],
        }

        self.file_path = fpath
        if fpath is not None:
            self.parse()

    def filter(self, iterable, method="exclusive", bag="unigram"):
        """ filter a bag """
        if bag == "unigram":
            self.unigram_bag = self.unigram_bag.filter(iterable, method=method)
        elif bag == "bigram":
            self.unigram_bag = self.unigram_bag.filter(iterable, method=method)
        else:
            print("Invalid bag")

    def analysis(self, compare, stopwords):
        """ analyze the bag based on the another bag/class
            compare: the bag that we wish to compare to/with
         """
        unigram_uniques = self.unigram_bag.filter(self.unigram_bag.merge(compare.unigram_bag, method="subtract"),
                                                  method="inclusive")
        bigram_uniques = self.bigram_bag.filter(self.bigram_bag.merge(compare.bigram_bag, method="subtract"),
                                                method="inclusive")
        stopword_uniques = unigram_uniques.filter(stopwords, method="inclusive")
        stopword_limit = 10 if len(stopword_uniques) >= 10 else len(stopword_uniques)

        self.result["unigram top 10"] = unigram_uniques.ordered(key=lambda x: x.score if x.score != 0 else -999)[:10]
        self.result["bigram top 10"] = bigram_uniques.ordered(key=lambda x: x.score if x.score != 0 else -999)[:10]
        self.result["stopword top 10"] = stopword_uniques.ordered(key=lambda x: x.score
                                                                  if x.score != 0 else -999)[:stopword_limit]

    def add_document(self, words, bag="unigram"):
        """ increment existing document count of given list words (only for unigram) """
        if bag == "unigram":
            for word in words:
                self.unigram_bag[word].existing_document_count += 1
        elif bag == "bigram":
            for word in words:
                self.bigram_bag[word].existing_document_count += 1
        else:
            print("Invalid bag")

    def parse_unigram(self, line):
        """ add unigram words from a line to unigram bag """
        # prevent repetition of the word in the line, for document count variable
        unique_set = set()
        for word in line.split(" "):
            self.unigram_bag.add(word)
            unique_set.add(word)

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
                clean_line = "{} {} {}".format(self.line_prepend, line.lower().strip(), self.line_append)
                self.parse_unigram(clean_line)
                self.parse_bigram(clean_line)

                self.line_count += 1

        self.unigram_bag.calculate_weights(number_of_documents=2)
        self.bigram_bag.calculate_weights(number_of_documents=2)
