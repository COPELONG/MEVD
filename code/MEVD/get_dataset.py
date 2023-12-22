
import os
import pandas
from vectorize_fragment import FragmentVectorizer
from Parser import parameter_parser
args = parameter_parser()

for arg in vars(args):
    print(arg, getattr(args, arg))


def parse_file(filename):
    """
    读取处理好的文件，主要是将train_data的文件中的对应中的文件进行处理
    :param filename: 文件名
    :return:
    """
    with open(filename, "r", encoding="utf8") as file:
        fragment = []
        fragment_val = 0
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if "-" * 33 in line and fragment:
                yield fragment, fragment_val
                fragment = []
            elif stripped.split()[0].isdigit():
                if fragment:
                    if stripped.isdigit():
                        fragment_val = int(stripped)
                    else:
                        fragment.append(stripped)
            else:
                fragment.append(stripped)
    return fragment



"""
假设所有碎片都可以放入内存，构建碎片字典列表字典包含碎片和漏洞指示器将每个碎片添
加到碎片向量器训练碎片向量器模型，通过片段列表再次准备向量化循环将
每个片段向量化，并将向量放入新的列表中。在处理所有片段时，将字典列表转换为数据帧
"""


def get_vectors_df_1(filename, vector_length=300, embedding=args.embedding):
    embedding = "word2vec"
    fragments = []
    count = 0
    vectorizer = FragmentVectorizer(vector_length)
    for fragment, val in parse_file(filename):
        count += 1
        print("Collecting fragments...", count, end="\r")
        vectorizer.add_fragment(fragment)
        row = {"fragment": fragment, "val": val}
        fragments.append(row)
    print('Found {} forward slices and {} backward slices'
          .format(vectorizer.forward_slices, vectorizer.backward_slices))
    print()
    print("Training model...", end="\r")
    if embedding == "word2vec":
        vectorizer.train_word2vec_model()
    vectors = []
    count = 0
    for fragment in fragments:
        count += 1
        print("Processing fragments...", count, end="\r")
        vector = vectorizer.vectorize(fragment["fragment"])
        row = {"vector": vector, "val": fragment["val"]}
        vectors.append(row)
    print()
    df = pandas.DataFrame(vectors)
    return df





def get_vectors_df_2(filename, vector_length=300, embedding=args.embedding):
    embedding = "FastText"
    fragments = []
    count = 0
    vectorizer = FragmentVectorizer(vector_length)
    for fragment, val in parse_file(filename):
        count += 1
        print("Collecting fragments...", count, end="\r")
        vectorizer.add_fragment(fragment)
        row = {"fragment": fragment, "val": val}
        fragments.append(row)
    print('Found {} forward slices and {} backward slices'
          .format(vectorizer.forward_slices, vectorizer.backward_slices))
    print()
    print("Training model...", end="\r")
    if embedding == "FastText":
        vectorizer.train_FastText_model()
    vectors = []
    count = 0
    for fragment in fragments:
        count += 1
        print("Processing fragments...", count, end="\r")
        vector = vectorizer.vectorize(fragment["fragment"])
        row = {"vector": vector, "val": fragment["val"]}
        vectors.append(row)
    print()
    df = pandas.DataFrame(vectors)
    return df

def get_vectors_df_3(filename, vector_length=300, embedding=args.embedding):
    embedding = "Doc2Vec"
    fragments = []
    count = 0
    vectorizer = FragmentVectorizer(vector_length)
    for fragment, val in parse_file(filename):
        count += 1
        print("Collecting fragments...", count, end="\r")
        vectorizer.add_fragment(fragment)
        row = {"fragment": fragment, "val": val}
        fragments.append(row)
    print('Found {} forward slices and {} backward slices'
          .format(vectorizer.forward_slices, vectorizer.backward_slices))
    print()
    print("Training model...", end="\r")
    if embedding == "Doc2Vec":
        vectorizer.train_Doc2vec_model()
    vectors = []
    count = 0
    for fragment in fragments:
        count += 1
        print("Processing fragments...", count, end="\r")
        vector = vectorizer.vectorize(fragment["fragment"])
        row = {"vector": vector, "val": fragment["val"]}
        vectors.append(row)
    print()
    df = pandas.DataFrame(vectors)
    return df

def get_df_1():
    embedding = "word2vec"
    filename = args.dataset
    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_" + embedding + "_fragment_vectors.pkl"
    vector_length = args.vector_dim
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df_1(filename, vector_length)
        df.to_pickle(vector_filename)
    return df, base


def get_df_2():
    embedding = "FastText"
    filename = args.dataset
    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_" + embedding + "_fragment_vectors.pkl"
    vector_length = args.vector_dim
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df_2(filename, vector_length)
        df.to_pickle(vector_filename)
    return df, base

def get_df_3():
    embedding = "Doc2vec"
    filename = args.dataset
    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_" + embedding + "_fragment_vectors.pkl"
    vector_length = args.vector_dim
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df_3(filename, vector_length)
        df.to_pickle(vector_filename)
    return df, base









