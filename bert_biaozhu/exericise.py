from process_examples import segment
inputfile = "H:/conpetition/competition_2/BERT/BERT_CIXINGBIAOZHU_DATA/train.txt"
lines = []

with open(inputfile, "r", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip().split("\t")
        if len(line) != 2:
            continue
        #print(line)
        words = line[0]
        labels = line[1]
        if len(words) > 128-2:
            words_list, labels_list = segment(words, labels)
            for each_words, each_labels in zip(words_list, labels_list):
                lines.append([each_words, each_labels])
        else:
            lines.append([words, labels])
    print("lines:{}".format(lines))


