import glob
import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from IPython.display import display
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

pd.options.display.width = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
pd.set_option('expand_frame_repr', False);

############################# CELL ######################################################################################

DATA_PATH = "data"
#BASE_MODEL = "huggingface_local/codebert-base"
BASE_MODEL = "microsoft/codebert-base"
N_SPLITS = 5
SEQ_LEN = 256
RANDOM_STATE = 42
NO_EPOCHS = 100
BATCH_SIZE = 64


def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
            .assign(id=os.path.splitext(os.path.basename(path))[0])
            .rename_axis('cell_id')
    )

###################################### CELL #####################################################################################
class DataSetGenerator:
    def __init__(self, data_path, order_file_path, seq_len, drop_rate=0.9, mode= 'train', nb_limit=None):
        self.data_path = data_path
        self.order_file_path = order_file_path
        self.seq_len = seq_len
        self.nb_limit = nb_limit
        self.mode = mode
        self.drop_rate = drop_rate
        self.df_orders = None

    def get_ranks(self, base, derived):
            return [base.index(d) for d in derived]

    def load_train_orders(self, df):
        # Split the string representation of cell_ids into a list
        self.df_orders = pd.read_csv(self.order_file_path, index_col='id').squeeze("columns").str.split()
        df_orders_ = self.df_orders.to_frame().join(df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
                                                    how='right')

        ranks = {}
        for id_, cell_order, cell_id in df_orders_.itertuples():
            ranks[id_] = {'cell_id': cell_id, 'rank': self.get_ranks(cell_order, cell_id)}

        return (
            pd.DataFrame
                .from_dict(ranks, orient='index')
                .rename_axis('id')
                .apply(pd.Series.explode)
                .set_index('cell_id', append=True)
        )

    def load_data(self):
        paths = glob.glob(self.data_path)
        if self.nb_limit is not None:
            paths = paths[:self.nb_limit]

        notebooks = [read_notebook(path) for path in tqdm(paths, desc='Train NBs')]
        df = (pd.concat(notebooks)
                .set_index('id', append=True)
                .swaplevel()
                .sort_index(level='id', sort_remaining=False))

        if self.mode in ['train', 'val']:
            df = df.reset_index().merge( self.load_train_orders( df ), on=["id", "cell_id"])
        else:
            df.reset_index( inplace = True )

        df = df.dropna()
        #display(df)

        return df, dict(zip(df['cell_id'].values, df['source'].values))

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    # pair up any 2 cells and label the pair according to the order in the notebook
    def pair_and_label(self, df):
        triplets = []
        random_drop = np.random.random(size=10000) > self.drop_rate
        count = 0

        # for each notebook
        for id, df_tmp in tqdm(df.groupby('id')):
            df_tmp_md = df_tmp[df_tmp['cell_type'] == 'markdown']
            df_tmp_rank = df_tmp['rank'].values
            df_tmp_cid = df_tmp['cell_id'].values

            for md_cid, rank in df_tmp_md[['cell_id', 'rank']].values:
                # if the cell is right after this markdown cell 1, otherwise 0
                labels = np.array([(r == (rank + 1)) for r in df_tmp_rank]).astype('int')

                for cd_cid, label in zip(df_tmp_cid, labels):
                    count += 1
                    if label == 1 or self.mode in ['test','val'] or random_drop[count % 10000]:
                        triplets.append([md_cid, cd_cid, label])

        return triplets

    def cleanup_text(self, string):
        for char in ['\r\n', '\r', '\n']:
            string = string.replace(char, ' ')
        return string

    def tokenize_and_label(self, df, source_dict, tokenizer):
        triplets = self.pair_and_label(df)

        input_ids = np.zeros((len(triplets), self.seq_len), dtype="int32")
        attention_mask = np.zeros((len(triplets), self.seq_len), dtype="int32")
        segment_ids = np.zeros((len(triplets), self.seq_len), dtype="int32")
        labels = np.zeros((len(triplets)), dtype='int32')

        for i, x in enumerate(tqdm(triplets, total=len(triplets))):
            markdown_source = self.cleanup_text(source_dict[x[0]])
            code_source = self.cleanup_text(source_dict[x[1]])
            tokens_md = tokenizer.tokenize(markdown_source)[:self.seq_len]
            tokens_cd = tokenizer.tokenize(code_source)[:self.seq_len]

            # truncate the both tokens together to length less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self.truncate_seq_pair(tokens_md, tokens_cd, self.seq_len - 3)
            tokens = ['CLS'] + tokens_md + ['SEP'] + tokens_cd + ['SEP']
            seg_ids = [1] + [0] * (len(tokens_md) + 1) + [1] * (len(tokens_cd) + 1)

            input = tokenizer.convert_tokens_to_ids(tokens)
            a_mask = [1] * len(input)
            padding_length = self.seq_len - len(input)

            # padd 0 to the right
            input += ([0] * padding_length)
            a_mask += ([0] * padding_length)
            seg_ids += ([0] * padding_length)
            input_ids[i] = input
            attention_mask[i] = a_mask
            segment_ids[i] = seg_ids
            labels[i] = x[2]

        return input_ids, attention_mask, segment_ids, labels

    def generate_dataset(self, tokenizer, batch_size, ordered=False, repeated=False ):

        df, source_dict = self.load_data()
        input_ids, attention_mask, segment_ids, labels = self.tokenize_and_label(df, source_dict, tokenizer)
        # Return batched and prefetched dataset
        if self.mode == 'train':
            dataset = tf.data.Dataset.from_tensor_slices(
                ({"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": segment_ids}, labels)
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": segment_ids}
            )
        if repeated:
            dataset = dataset.repeat()
        if not ordered:
            dataset = dataset.shuffle(1024)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset



 class CodeBERTModel(tf.keras.Model):
    def __init__(self, base_model):
      super(CodeBERTModel, self).__init__(name="prediction")
      self.base_encoder = transformers.TFRobertaModel.from_pretrained(base_model)
      self.dropout = tf.keras.layers.Dropout(0.1)
      self.dense = tf.keras.layers.Dense(1, activation="linear", dtype="float32")

    def call(self, inputs):
      base_encoder_inputs = { "input_ids": inputs[0], "attention_mask": inputs[1], "token_type_ids": inputs[2]}
      encoder_outputs = self.encoder( base_encoder_inputs )
      pooled_output = encoder_outputs[1] # is the pooled output from transformers BERT output
      x = self.dropout(pooled_output)
      x = self.dense(x)
      return x


def get_model() -> tf.keras.Model:
    backbone = transformers.TFRobertaModel.from_pretrained(BASE_MODEL)
    input_ids = tf.keras.layers.Input(
        shape=(SEQ_LEN,),
        dtype=tf.int32,
        name="input_ids",
    )
    attention_mask = tf.keras.layers.Input(
        shape=(SEQ_LEN,),
        dtype=tf.int32,
        name="attention_mask",
    )
    input_type_ids = tf.keras.layers.Input(
        shape=(SEQ_LEN,),
        dtype=tf.int32,
        name="token_type_ids",
    )
    x = backbone(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": input_type_ids
        },
    )
    outputs = tf.keras.layers.Dense(1, activation="linear", dtype="float32")(x[0][:, 0, :])

    model = tf.keras.Model(
        inputs=[input_ids, attention_mask, input_type_ids],
        outputs=outputs,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    return model

###################################### CELL #####################################################################################
from bisect import bisect

class Inferor:
    def __init__(self, data_path, model_path ):
        self.d_path = data_path
        self.m_path = model_path

    @staticmethod
    def count_inversions(a):
        inversions = 0
        sorted_so_far = []
        for i, u in enumerate(a):
            j = bisect(sorted_so_far, u)
            inversions += i - j
            sorted_so_far.insert(j, u)
        return inversions

    @staticmethod
    def kendall_tau(ground_truth, predictions):
        total_inversions = 0
        total_2max = 0  # twice the maximum possible inversions across all instances
        for gt, pred in zip(ground_truth, predictions):
            ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
            total_inversions += Inferor.count_inversions(ranks)
            n = len(gt)
            total_2max += n * (n - 1)
        return 1 - 4 * total_inversions / total_2max

    # this will sort an array based on relative pair wise order
    # assuming only the bottom portion of the original list needs to be sorted
    # the relative rank is given as one-hot-index of the after index, i.e  1 means the index is after the index being sorted 0, means otherwise
    @staticmethod
    def sort_with_pairwise_rank(  orig_order, relative_rank):
        no_elements = len(orig_order)

        no_to_sort = int(len( relative_rank)/no_elements)
        fix_elements = no_elements - no_to_sort
        new_order = np.copy(orig_order)

        # the list of cells that right after the index,
        buddy_lst = [[]]*no_to_sort
        for i in range(no_to_sort):
            tmp_rel_rank = relative_rank[i * no_elements:(i + 1) * no_elements]
            i_plus_1 = np.argmax(tmp_rel_rank)
            buddy_lst[i] = [i_plus_1]

        # need to take care of nested buddy list, i.e., we can have [12,17] and [17,10]
        for k, lst_1 in enumerate(buddy_lst):
            for lst in buddy_lst:
                if k+fix_elements in lst:
                    lst.extend(lst_1)

        for index, lst in enumerate( buddy_lst):
            new_order[index + fix_elements ] = min(lst) + 0.5 * 0.9**(len(lst))

        return  new_order

    def get_ordered_cells_from_pred(self, df, preds):
        count = 0
        for id, df_tmp in tqdm(df.groupby('id')):
            df_tmp_md = df_tmp[df_tmp['cell_type']=='markdown']

             #original order of all cells, markdown and code in a NB
            orig_order = df_tmp['pred'].values
            no_cell = len(orig_order)
            no_md = len(df_tmp_md)

            preds_tmp = preds[count:count+no_md * no_cell]

            count += no_md * no_cell
            new_order = Inferor.sort_with_pairwise_rank(orig_order, preds_tmp )
            df.loc[test_df['id'] == id, 'pred'] = new_order

        # extract the cells in the new order
        y = df.reset_index().sort_values('pred').groupby('id')['cell_id'].apply(list)
        return y

#################################### CELL inference ######################################################################
#paths = glob.glob(os.path.join(DATA_PATH, "val", "*.json"))
tokenizer = transformers.RobertaTokenizer.from_pretrained(BASE_MODEL, do_lower_case=True)

val_ds_generator = DataSetGenerator(os.path.join(DATA_PATH, 'val', '*.json'),
                                    os.path.join(DATA_PATH, 'train_orders.csv'), SEQ_LEN, mode='val')

val_df, source_dict = val_ds_generator.load_data()

input_ids, attention_mask, segment_ids, labels = val_ds_generator.tokenize_and_label(val_df, source_dict, tokenizer)
display(val_df)

test_ds_generator = DataSetGenerator(os.path.join(DATA_PATH, "val", "*.json"), os.path.join(DATA_PATH, 'train_orders.csv'), SEQ_LEN, mode='test')

test_df, _ = test_ds_generator.load_data()

test_df["rank"] = test_df.groupby(["id"]).cumcount()
test_df["pred"] = test_df.groupby(["id"])["rank"].rank(pct=False)


# val_dataset = get_dataset(
#     input_ids=input_ids,
#     attention_mask=attention_mask,
#     segment_ids = segment_ids,
#     ordered=True,
# )
#model = get_model()
#model.load_weights("model_0.h5")
#y_pred = model.predict(val_dataset)

preds_copy = labels #y_pred

######################################### calculate Kendal_tau ##########################################################
inferor = Inferor(DATA_PATH, None)
y_dummy = inferor.get_ordered_cells_from_pred(test_df, preds_copy)
k = Inferor.kendall_tau(val_ds_generator.df_orders.loc[y_dummy.index], y_dummy)
print(f"kendall_tau = {k}")
exit()
############################################ submit #####################################################################

test_df = test_df.reset_index().sort_values("pred").groupby("id", as_index=False)["cell_id"].apply(lambda x: " ".join(x))
test_df.rename(columns={"cell_id": "cell_order"}, inplace=True)
test_df.to_csv("submission.csv", index=False)
display(test_df)