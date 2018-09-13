import os
import json
from sklearn.utils import shuffle


def create_label_set():

    raw_data = 'input/intent.train_data.2w'
    with open(raw_data, 'r', encoding='utf-8') as f:
        content = f.readlines()
    lines = [x.strip() for x in content]

    corpora = []
    labels = []
    label_dict = dict()
    for line in lines:
        json_string = line.split("	")[-1]
        json_obj = json.loads(json_string)
        intents = json_obj["intents"]
        label = []
        for intent in intents:
            intent_val = intent["target"]["value"]
            if intent_val:
                label.append(intent_val)
                if intent_val in label_dict:
                    label_dict[intent_val] += 1
                else:
                    label_dict[intent_val] = 1
        if not label:
            continue
        labels.append(label)
        corpora.append(json_obj["sentence"])

    label_path = 'output/labels.txt'
    with open(label_path, 'w', encoding='utf-8') as f:
        for key in label_dict:
            f.write(key + "\t" + str(label_dict[key]) + "\n")

    return labels, corpora


def create_data_set(path, labels, corpora):
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(len(labels)):
            f.write("、".join(labels[i]) + "\t" + corpora[i] + "\n")


def split_data_set(labels, corpora, split_ratio=0.8):
    labels, corpora = shuffle(labels, corpora)
    split_index = int(len(labels) * split_ratio)
    labels_1 = labels[:split_index]
    corpora_1 = corpora[:split_index]
    labels_2 = labels[split_index:]
    corpora_2 = corpora[split_index:]
    return labels_1, corpora_1, labels_2, corpora_2


def create_split_data_sets(labels, corpora):

    base_dir = 'output'
    train_dir = os.path.join(base_dir, 'train.txt')
    test_dir = os.path.join(base_dir, 'test.txt')
    val_dir = os.path.join(base_dir, 'val.txt')

    # 将数据分割为训练集与测试集
    labels_train, corpora_train, labels_test, corpora_test = split_data_set(labels, corpora)
    # 保存测试集数据
    create_data_set(test_dir, labels_test, corpora_test)
    # 将训练集进一步分割为训练集与验证集
    labels_train, corpora_train, labels_val, corpora_val = split_data_set(labels_train, corpora_train)
    # 保存训练集以及验证集数据
    create_data_set(train_dir, labels_train, corpora_train)
    create_data_set(val_dir, labels_val, corpora_val)


def main():
    labels, corpora = create_label_set()
    create_split_data_sets(labels, corpora)


if __name__ == '__main__':
    main()
