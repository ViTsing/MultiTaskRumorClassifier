import os
import numpy as np
import json
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from treeLabels import tree2branch
from convert_veracity_annotations import convert_annotations

input_dir = 'all-rnr-annotated-threads/'
output_dir = 'output_files/'
count_tweets = 0
# rumor threads
count_threads = 0
arr_threads = list()

train = ['ferguson',
         'ottawashooting',
         'sydneysiege',
         'charliehebdo',
         'germanwings-crash']


class Embeddings:
    def __init__(self):
        self.glove_input_file = 'glove.twitter/glove.twitter.27B.200d.txt'
        self.word2vec_output_file = 'glove.twitter/glove.twitter.27B.200d.word2vec.txt'
        self.glove_model = None

    def tweet_embedding(self, text):
        """
        :param text: str, a piece of tweet
        :return:
        """
        tweet_vec_list = list()
        embeddings = np.ndarray(shape=(200,))
        text = "".join(filter(str.isalnum, text))
        count = 0
        for word in text:
            word = word.lower()
            if word in self.glove_model:
                em_word = self.glove_model[word]
                tweet_vec_list.append(em_word)
                try:
                    embeddings += em_word
                    count += 1
                except ValueError:
                    print(em_word)
                    continue
        if count > 0:
            embeddings = embeddings / count
        return embeddings

    def get_word2vec(self):
        if not os.path.exists(self.word2vec_output_file):
            print('not exist')
            (count, dimensions) = glove2word2vec(self.glove_input_file, self.word2vec_output_file)
            print(count, dimensions)
        print('loading w2v')
        self.glove_model = KeyedVectors.load_word2vec_format(self.word2vec_output_file, binary=False)
        print('w2v model loaded!!! ')


if __name__ == "__main__":

    debug = False
    if debug is False:
        w2v = Embeddings()
        w2v.get_word2vec()

    for root, all_news, files in os.walk(input_dir):
        # walk the first-level dirs [different news]
        news_idx = 0
        for news in all_news:
            thread_dirs = os.path.join(input_dir, news) + '/rumours/'
            news_list = list()
            veracity_embs = list()
            for _root, _dirs, _files in os.walk(thread_dirs):
                # walk the third-level dirs [all threads with numbers as folder names]
                for thread_dir in _dirs:
                    thread_id = thread_dir
                    thread_list = list()
                    threads_emb = dict()
                    count_threads += 1

                    source = os.path.join(thread_dirs, thread_dir) + '/source-tweets'
                    for _, _, sources in os.walk(source):
                        for source_file in sources:
                            if source_file[0] != '.':
                                with open(os.path.join(source, source_file), 'r') as f:
                                    # print(os.path.join(reactions, retweet))
                                    json_content = json.load(f)
                                    text = json_content['text']
                                    if debug is False:
                                        tweet_emb = w2v.tweet_embedding(text)
                                    else:
                                        tweet_emb = [1, 2]
                                    source_file = source_file[0:-5]
                                    threads_emb[source_file] = tweet_emb
                                    count_tweets += 1

                    reactions = os.path.join(thread_dirs, thread_dir) + '/reactions/'
                    for _, _, retweets in os.walk(reactions):
                        for reaction_file in retweets:
                            if reaction_file[0] != '.':
                                with open(os.path.join(reactions, reaction_file), 'r') as f:
                                    # print(os.path.join(reactions, retweet))
                                    json_content = json.load(f)
                                    text = json_content['text']
                                    if debug is False:
                                        tweet_emb = w2v.tweet_embedding(text)
                                    else:
                                        tweet_emb = [1, 2]
                                    reaction_file = reaction_file[0:-5]
                                    threads_emb[reaction_file] = tweet_emb
                                    count_tweets += 1

                    with open(os.path.join(thread_dirs, thread_dir) + '/annotation.json', 'r') as f:
                        annotation_json_content = json.load(f)
                        veracity = convert_annotations(annotation_json_content, string=False)
                        veracity_emb = np.zeros(3)
                        veracity_emb[veracity] = 1

                    structure_file = os.path.join(thread_dirs, thread_dir) + '/structure.json'
                    t = tree2branch(structure_file, thread_id)
                    for branch in t:
                        branch_list = list()
                        if len(branch) > 0:
                            for tweet_idx in branch:
                                if tweet_idx in threads_emb.keys():
                                    branch_list.append(threads_emb[tweet_idx])
                            thread_list.append(branch_list)
                            veracity_embs.append(veracity_emb)
                    news_list += thread_list
                break

            # context embeddings
            train_news_array = news_list
            # tracking labels for each thread
            tracking_label = np.zeros(len(all_news))
            tracking_label[news_idx] = 1
            tracking_labels = np.expand_dims(tracking_label, axis=0)
            tracking_labels = np.repeat(tracking_labels, len(train_news_array), axis=0)
            # veracity labels for each thread
            veracity_labels = np.array(veracity_embs)

            _output_dir = os.path.join(output_dir, news)
            if not os.path.exists(_output_dir):
                os.makedirs(_output_dir)
                print('make dir: ' + _output_dir)

            np.save(_output_dir + '/train_arrays.npy', train_news_array)
            np.save(_output_dir + '/tracking_labels.npy', tracking_labels)
            np.save(_output_dir + '/veracity_labels.npy', veracity_labels)

            news_idx += 1
        break
