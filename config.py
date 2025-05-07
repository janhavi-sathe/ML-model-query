import os
import logging

class Config:
    EXPERIMENT_GROUP = True
    TEXT_DATA_FILENAME = 'snips_bert-mini_test_ranked.csv'
    TEXT_SIMILARITY_FILENAME = 'bert-mini-sim_matrix.npy'

    TEXT_ID2LABEL = {0: 'AddToPlaylist', 1: 'BookRestaurant', 2: 'GetWeather', 3: 'PlayMusic', 4: 'RateBook', 5: 'SearchCreativeWork', 6: 'SearchScreeningEvent'}
    N_TEXT_CLASSES = 7

    RLA_SELECTED_TEXTS = [181, 54, 131, 133, 77, 159, 137, 125, 128, 103, 7, 153, 136, 190, 108, 66, 148, 6, 132, 138, 15, 63, 35, 38, 67, 91, 116, 32, 62, 161, 107, 83, 143, 41, 118, 162, 31, 163, 191, 130, 170, 79, 94, 80, 147, 39, 183, 113, 100, 196]

    TOP_N_SIMILAR_TEXTS = 10
    TEXT_SIMILARITY_THRESHOLD = 0.9

    DEFAULT_PER_PAGE = 8 

class LoggerConfig:
    foldername = "logs/experiment" if Config.EXPERIMENT_GROUP else "logs/control"
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), foldername)
    encoding = "utf-8"
    filemode = "a"
    format = "{asctime} - {levelname} - {message}"
    style = "{"
    datefmt = "%Y-%m-%d %H:%M"
    level = logging.INFO
