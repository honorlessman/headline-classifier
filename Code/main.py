from Code.data import Data
from Code.line import parse_line
from Code.naive_bayes import NaiveBayes
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

REAL_TRAIN_DATA_PATH = "data/clean_real-Train.txt"
FAKE_TRAIN_DATA_PATH = "data/clean_fake-Train.txt"

TEST_DATA_PATH = "data/test/test.txt"
TEST_PATH = "data/test.csv"

LINE_PREPEND = "<l>"
LINE_APPEND = "</l>"

FILTER_STOPWORDS = False


if __name__ == "__main__":
    # parse data from file
    real_data = Data(fpath=REAL_TRAIN_DATA_PATH, line_prepend=LINE_PREPEND, line_append=LINE_APPEND)
    fake_data = Data(fpath=FAKE_TRAIN_DATA_PATH, line_prepend=LINE_PREPEND, line_append=LINE_APPEND)
    test_data = parse_line(TEST_PATH, line_prepend=LINE_PREPEND, line_append=LINE_APPEND)

    # filter out stopwords
    if FILTER_STOPWORDS:
        real_data.filter(ENGLISH_STOP_WORDS, method="exclusive")
        fake_data.filter(ENGLISH_STOP_WORDS, method="exclusive")
        for line in test_data:
            line.filter(ENGLISH_STOP_WORDS, method="exclusive")

    # apply naive bayes
    model = NaiveBayes([fake_data, real_data])
    model.fit(test_data)

    # get results
    print("Unigram accuracy score: {}".format(model.accuracy_score))
    print("Bigram accuracy score: {}".format(model.bi_accuracy_score))

    print("done")
