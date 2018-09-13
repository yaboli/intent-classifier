import os
from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr
import jieba
from sklearn.preprocessing import MultiLabelBinarizer
import pickle


def read_obj(path):  # 读取对象函数
    file_obj = open(path, "rb")
    obj = pickle.load(file_obj)  # 使用pickle.load反序列化对象
    file_obj.close()
    return obj


def write_obj(path, obj):  # 写入对象函数
    file_obj = open(path, "wb")
    pickle.dump(obj, file_obj)  # 持久化对象
    file_obj.close()


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename, cut_word=True):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    # 如果cut_word为True，语料的最小单位为词汇；否则为中文字符
                    if cut_word:
                        segs = jieba.cut(content)
                        contents.append(" ".join(segs).split(" "))
                    else:
                        contents.append(list(content))
                    labels.append(label)
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                break
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir, cut_word=False)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    # 如果语料中的词汇量小于vocab_size，则使用全部词汇
    vocab_in_content = len(counter.keys())
    if vocab_in_content < vocab_size:
        vocab_size = vocab_in_content
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    with open_file(vocab_dir) as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def process_file(filename, word_to_id, max_length=25):
    """将文件转换为id表示"""
    contents, labels = read_file(filename, cut_word=False)

    data_id, label_tup = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_tup.append(labels[i].split("、"))

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)

    # 使用sklearn.preprocessing的MultiLabelBinarizer为标签集作one-hot encoding
    base_dir = 'resources'
    one_hot_dir = os.path.join(base_dir, 'one_hot_encoder.pkl')
    # 检查encoder是否存在
    if not os.path.exists(one_hot_dir):
        # 为标签输出作one-hot encoding
        one_hot_encoder = MultiLabelBinarizer()
        y_pad = one_hot_encoder.fit_transform(label_tup)
        # 保存one-hot encoder
        write_obj(one_hot_dir, one_hot_encoder)
    else:
        one_hot_encoder = read_obj(one_hot_dir)
        y_pad = one_hot_encoder.transform(label_tup)

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def main():
    base_dir = 'data'
    train_dir = os.path.join(base_dir, 'train.txt')
    vocab_dir = os.path.join(base_dir, 'vocab.txt')

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir)

    # # One-hot encoding example
    # y = [['Texas', 'Florida'],
    #      ['California'],
    #      ['Texas', 'Florida'],
    #      ['Delware'],
    #      ['Alabama']]
    # one_hot = MultiLabelBinarizer()
    # arr = one_hot.fit_transform(y)
    # print(arr)
    # print(one_hot.classes_)


if __name__ == '__main__':
    main()
