from Code.data_parser import Parser

REAL_TRAIN_DATA_PATH = "data/clean_real-Train.txt"
FAKE_TRAIN_DATA_PATH = "data/clean_fake-Train.txt"

if __name__ == "__main__":
    real_data = Parser(REAL_TRAIN_DATA_PATH)
    fake_data = Parser(FAKE_TRAIN_DATA_PATH)

    sorted_real_data = real_data.sorted_bag
    sorted_fake_data = fake_data.sorted_bag

    print("done")
