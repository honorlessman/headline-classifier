import csv
from Code.data import Data


class Line(Data):
    """ Line object for test data, child class of data """
    def __init__(self, line, category, line_prepend="", line_append=""):
        super().__init__(line_prepend=line_prepend, line_append=line_append)

        self.line = "{}{}{}".format(self.line_prepend, line.lower().strip(), self.line_append)
        self.category = category

        self.fake_score = 0
        self.real_score = 0
        self.is_fake = -1

        self.bi_fake_score = 0
        self.bi_real_score = 0
        self.bi_is_fake = -1

        # start parsing
        self.parse()

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

    def parse(self):
        """ parse line into bags """
        self.parse_bigram(self.line)
        self.parse_unigram(self.line)


def parse_line(file, line_append="", line_prepend=""):
    """ parse csv into lines,
     does not fix anything on csv so fix it yourself, be responsible """
    f = open(file)
    reader = csv.reader(f, delimiter=',')

    # skip the header
    next(reader, None)

    # read rows into lines
    out = [Line(row[0], 1 if row[1] == "fake" else 0,
                line_append=line_append, line_prepend=line_prepend) for row in reader]
    f.close()

    return out
