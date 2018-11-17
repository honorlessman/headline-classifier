from Code.data_parser import Parser
from Code.line import Line
from Code.naive_bayes import NaiveBayes
import csv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

REAL_TRAIN_DATA_PATH = "data/clean_real-Train.txt"
FAKE_TRAIN_DATA_PATH = "data/clean_fake-Train.txt"

# REAL_TRAIN_DATA_PATH = "data/test/new_reals.txt"
# FAKE_TRAIN_DATA_PATH = "data/test/new_fakes.txt"

TEST_DATA_PATH = "data/test/test.txt"
TEST_PATH = "data/test.csv"


def line_parser(path):
    """ parse csv into lines,
     does not fix anything on csv so fix it yourself, be responsible """
    f = open(path)
    reader = csv.reader(f, delimiter=',')

    # skip the header
    next(reader, None)

    # read rows into lines
    out = [Line(row[0], 1 if row[1] == "fake" else 0) for row in reader]
    f.close()

    return out


if __name__ == "__main__":
    # parse data from file
    real_data = Parser(REAL_TRAIN_DATA_PATH)
    fake_data = Parser(FAKE_TRAIN_DATA_PATH)
    test_data = line_parser(TEST_PATH)

    # filter out stopwords
    real_data.filter(ENGLISH_STOP_WORDS, method="inclusive")
    fake_data.filter(ENGLISH_STOP_WORDS, method="inclusive")
    for line in test_data:
        line.filter(ENGLISH_STOP_WORDS, method="inclusive")

    # apply naive bayes
    model = NaiveBayes([fake_data, real_data])
    model.fit(test_data)

    # get results
    print("Unigram accuracy score: {}".format(model.accuracy_score))
    print("Bigram accuracy score: {}".format(model.bi_accuracy_score))

    print("done")
