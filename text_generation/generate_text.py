from sklearn.utils import shuffle


# 句式 + 标签词 -> 填词造句
def generate_corpora(label_path, corpora_uncut_path):
    pattern_path = 'text_generation/resources/句式.txt'
    # 载入句式
    with open(pattern_path, 'r', encoding='utf-8') as f:
        patterns = f.readlines()
    patterns = [x.strip() for x in patterns]

    # 载入标签词
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    labels = [x.strip() for x in labels]

    # 将标签词逐个填入句式
    with open(corpora_uncut_path, 'w', encoding='utf-8') as f:
        for pattern in patterns:
            for label in labels:
                line = pattern.replace("{}", label)
                line += "\n"
                f.write(line)


def generate_all_corpora(dir1, dir2, label_files, corpora_files):
    for i in range(len(label_files)):
        label_path = dir1 + label_files[i]
        corpora_path = dir2 + corpora_files[i]
        generate_corpora(label_path, corpora_path)


def create_data_sets(directory, corpora_files):
    train_path = directory + 'train.txt'
    val_path = directory + 'val.txt'

    for corpora_file in corpora_files:
        label = corpora_file.replace('_corpora.txt', '')
        full_path = directory + corpora_file
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        # shuffle and split corpora
        content = shuffle(content)
        split_ratio = 0.7
        split_index = int(len(content) * split_ratio)
        train_corpora = content[:split_index]
        val_corpora = content[split_index:]

        with open(train_path, 'a', encoding='utf-8') as f:
            for corpus in train_corpora:
                line = label + '\t' + corpus + '\n'
                f.write(line)

        with open(val_path, 'a', encoding='utf-8') as f:
            for corpus in val_corpora:
                line = label + '\t' + corpus + '\n'
                f.write(line)


def main():
    dir1 = 'text_generation/resources/'
    dir2 = 'text_generation/output/'

    label_files = ['责任主体标签.txt', '责任构成标签.txt', '减轻或免责事由标签.txt', '责任方式标签.txt', '诉讼程序问题标签.txt']
    corpora_files = ['责任主体_corpora.txt', '责任构成_corpora.txt', '减轻或免责事由_corpora.txt', '责任方式_corpora.txt', '诉讼程序问题_corpora.txt']

    # # 生成五个标签对应的语料库
    # generate_all_corpora(dir1, dir2, label_files, corpora_files)

    # 生成训练与验证集语料库
    create_data_sets(dir2, corpora_files)


if __name__ == '__main__':
    main()
