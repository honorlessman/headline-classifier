from Code.data import Data
from Code.line import parse_line
from Code.naive_bayes import NaiveBayes
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

REAL_TRAIN_DATA_PATH = "data/clean_real-Train.txt"
FAKE_TRAIN_DATA_PATH = "data/clean_fake-Train.txt"

TEST_PATH_PATH = "data/test.csv"

LINE_PREPEND = "<l>"
LINE_APPEND = "</l>"

FILTER_STOPWORDS = False
EXPERIMENTAL_FILTERING = False
TF_IDF = False
UNIGRAM_TOKEN_CLEAR = False

LIST_TOP_10 = True


if __name__ == "__main__":
    # parse data from file
    real_data = Data(fpath=REAL_TRAIN_DATA_PATH, line_prepend=LINE_PREPEND, line_append=LINE_APPEND)
    fake_data = Data(fpath=FAKE_TRAIN_DATA_PATH, line_prepend=LINE_PREPEND, line_append=LINE_APPEND)
    test_data = parse_line(TEST_PATH_PATH, line_prepend=LINE_PREPEND, line_append=LINE_APPEND)

    # update the document counts for unigram words, used to calculate tf-idf
    non_unique_unigram = real_data.unigram_bag.merge(fake_data.unigram_bag, method="intersect")
    non_unique_bigram = real_data.bigram_bag.merge(fake_data.bigram_bag, method="intersect")
    real_data.add_document(non_unique_bigram, bag="bigram")
    real_data.add_document(non_unique_unigram)
    fake_data.add_document(non_unique_bigram, bag="bigram")
    fake_data.add_document(non_unique_unigram)

    # filter out stopwords
    if FILTER_STOPWORDS:
        real_data.filter(ENGLISH_STOP_WORDS, method="exclusive")
        fake_data.filter(ENGLISH_STOP_WORDS, method="exclusive")
        for line in test_data:
            line.filter(ENGLISH_STOP_WORDS, method="exclusive")

    if EXPERIMENTAL_FILTERING:
        # EXPERIMENT, removal of words that exists in both data
        intersection = real_data.unigram_bag.merge(fake_data.unigram_bag, method="intersect")
        real_data.filter(intersection, method="exclusive")
        fake_data.filter(intersection, method="exclusive")

    if UNIGRAM_TOKEN_CLEAR:
        # clear the added token from unigram bags
        fake_data.filter([LINE_PREPEND.strip(), LINE_APPEND.strip()])
        real_data.filter([LINE_PREPEND.strip(), LINE_APPEND.strip()])
        for line in test_data:
            line.filter([LINE_PREPEND.strip(), LINE_APPEND.strip()])

    # apply naive bayes
    model = NaiveBayes([fake_data, real_data])
    model.fit(test_data, tf_idf=TF_IDF)

    # analyze the training data (check top 10s)
    real_data.analysis(fake_data, ENGLISH_STOP_WORDS)
    fake_data.analysis(real_data, ENGLISH_STOP_WORDS)

    # list the top10 words
    if LIST_TOP_10:
        print("\nTHE WORDS INDICATING THE HEADLINE IS FAKE")
        fake_data.pprint()
        print("\nTHE WORDS INDICATING THE HEADLINE IS REAL")
        real_data.pprint()
        print("")

    # get results
    print("Unigram accuracy: {:.2f}%".format(model.accuracy_score))
    print("Bigram accuracy: {:.2f}%".format(model.bi_accuracy_score))

    print("done")
