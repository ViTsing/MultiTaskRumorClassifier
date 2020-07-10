import numpy as np


def pad_list(lst):
    inner_max_len = max(map(len, lst))
    r_lst = [np.pad(l, ((0, inner_max_len - len(l)), (0, 0))) for l in lst if len(l) > 0]
    return np.array(r_lst)


_output_dir = "output_files/charliehebdo-all-rnr-threads/"
train_news_array = np.load(_output_dir + '/train_arrays.npy', allow_pickle=True)
padded_train_array = pad_list(train_news_array)
tracking_labels = np.load(_output_dir + '/tracking_labels.npy', allow_pickle=True)
veracity_labels = np.load(_output_dir + '/veracity_labels.npy', allow_pickle=True)

print(padded_train_array.shape)
print(tracking_labels.shape)
print(veracity_labels.shape)
