import os

class Config:
    TEXT_DATA_FILENAME = 'snips_bert-mini_test_ranked.csv'

    TEXT_ID2LABEL = {0: 'AddToPlaylist', 1: 'BookRestaurant', 2: 'GetWeather', 3: 'PlayMusic', 4: 'RateBook', 5: 'SearchCreativeWork', 6: 'SearchScreeningEvent'}
    N_TEXT_CLASSES = 7

    RLA_SELECTED_TEXTS = [181, 54, 131, 133, 77, 159, 137, 125, 128, 103, 7, 153, 136, 190, 108, 66, 148, 6, 132, 138, 15, 63, 35, 38, 67, 91, 116, 32, 62, 161, 107, 83, 143, 41, 118, 162, 31, 163, 191, 130, 170, 79, 94, 80, 147, 39, 183, 113, 100, 196]

    TOP_N_SIMILAR_TEXTS = 10
    TEXT_SIMILARITY_THRESHOLD = 0.9

    DEFAULT_PER_PAGE = 8 

