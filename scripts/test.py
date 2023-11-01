import numpy as np

def create_headers():
    headers = {}
    headers['cycle'] = [0]
    headers['ball'] = [1, 2, 3, 4, 5, 6, 7, 8]

    # 4 5 6, 7 8 9
    for i in range(1, 12):
        headers[f'tm-{i}-noise'] = list(range(9 + (i - 1) * 8, 9 + i * 8))  # max=4+11*3 = 37
        headers[f'opp-{i}-noise'] = list(range(97 + (i - 1) * 8, 97 + i * 8))  # max = 37+11*3 = 33+37 = 70
        headers[f'tm-{i}-full'] = list(range(8 + 185 + (i - 1) * 8, 8 + 185 + i * 8))  # max=70 + 33 = 103
        headers[f'opp-{i}-full'] = list(range(8 + 273 + (i - 1) * 8, 8 + 273 + i * 8))

    sub_headers = {
        'pos': [0, 1, 2],
        'vel': [3, 4, 5],
        'body': [6, 7]
    }
    return headers, sub_headers

def create_x_y_indexes(headers: dict[str, list[int]]):
    x_indexes = []
    y_indexes = []
    for key, value in headers.items():
        if key in ['cycle']:
            continue
        if key.find('full') != -1:
            continue
        x_indexes += value
        print('x', key)

    for key, value in headers.items():
        if key in ['cycle']:
            continue
        if key.find('noise') != -1:
            continue
        if key.find('ball') != -1:
            continue
        if key.find('tm') != -1:
            continue
        y_indexes += value[:2]
        print('y', key)
        

    return x_indexes, y_indexes

def test_labeled_y():
    opp_pos_noise = np.array(
        [[17, 18],
         [30, 23],
         [10, -10],
         [2, 3],]
    )
    opp_pos_full = np.array(
        [[10, 10],
         [10, 10],
         [10, 10],
         [10, 10],]
    )
    n_label = 20
    r = 20

    opp_pos_noise = np.where(opp_pos_noise == [-105, -105], [0, 0], opp_pos_noise)

    opp_err = opp_pos_noise - opp_pos_full
    opp_err = np.clip(opp_err, -r / 2, r / 2)
    index = np.floor((opp_err / r) * (n_label - 1)) + n_label / 2
    y_index = np.array(index[:, 0] * n_label + index[:, 1], dtype=np.uint32)

    y = np.zeros((opp_pos_noise.shape[0], n_label ** 2))
    y[np.arange(y_index.size), y_index] = 1

    for i in range(opp_pos_noise.shape[0]):
        print('-'*20)
        print(opp_pos_noise[i])
        print(opp_pos_full[i])
        print(opp_err[i])
        print(index[i])
        print(y_index[i])
        print(np.argmax(y[i]))
    return y


# h, _ = create_headers()

# x, y = create_x_y_indexes(h)

test_labeled_y()