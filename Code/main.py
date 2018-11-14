from Code.data_parser import Parser, Line
from Code.naive_bayes import NaiveBayes
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# REAL_TRAIN_DATA_PATH = "data/clean_real-Train.txt"
# FAKE_TRAIN_DATA_PATH = "data/clean_fake-Train.txt"

REAL_TRAIN_DATA_PATH = "data/test/new_reals.txt"
FAKE_TRAIN_DATA_PATH = "data/test/new_fakes.txt"

TEST_DATA_PATH = "data/test/test.txt"

if __name__ == "__main__":
    real_data = Parser(REAL_TRAIN_DATA_PATH)
    fake_data = Parser(FAKE_TRAIN_DATA_PATH)
    test_data = [Line(line) for line in open(TEST_DATA_PATH).readlines()]

    model = NaiveBayes([fake_data, real_data])
    model.fit(test_data)

    print("done")
