import numpy as np

max_seq_length = 128
def tough_seg(words, labels):
    words_list = []
    labels_list = []
    while len(words) > max_seq_length - 2:
        words_list.append(words[0:max_seq_length - 2])
        labels_list.append(labels[0:max_seq_length - 2])
        words = words[max_seq_length - 2:]
        labels = labels[max_seq_length - 2:]
    if len(words) > 0:
        words_list.append(words)

        labels_list.append(labels)
    return words_list, labels_list
def segment(words, labels):
    words_result_list = []
    labels_result_list = []
    print("max_words:{}".format(words))
    tmp = words.split("，")
    print("tmp:{}".format(tmp))
    start = 0
    for each in tmp:
        if len(each) == 0:
            start += 1
        elif len(each) <= 128 - 2:
            words_tmp_list, labels_tmp_list = tough_seg(each, labels[start: start + len(each)])
            print("words_tmp_list:{}".format(words_tmp_list))
            print("labels_tmp_list:{}".format(labels_tmp_list))
            words_result_list.extend(words_tmp_list)
            labels_result_list.extend(labels_tmp_list)
            start = start + len(each) + 1
        else:
            words_result_list.append(each)
            labels_result_list.append(labels[start: start + len(each)])
            start = start + len(each) + 1
    return words_result_list, labels_result_list
