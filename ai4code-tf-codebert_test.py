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
BASE_MODEL = "huggingface_local/codebert-base" #"microsoft/codebert-base"
N_SPLITS = 5
SEQ_LEN = 256
RANDOM_STATE = 42
NO_EPOCHS = 100

try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    STRATEGY = tf.distribute.experimental.TPUStrategy(TPU)
    BATCH_SIZE = 128 * STRATEGY.num_replicas_in_sync
except Exception:
    TPU = None
    #STRATEGY = tf.distribute.get_strategy()
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    BATCH_SIZE = 64
    NB_LIMIT = 10000

print("TensorFlow", tf.__version__)

if TPU is not None:
    print("Using TPU v3-8")
else:
    print("Using GPU/CPU")

###################################### CELL #####################################################################################
def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=os.path.splitext(os.path.basename(path))[0])
        .rename_axis('cell_id')
    )


def get_ranks(base, derived):
    return [base.index(d) for d in derived]


def truncate_seq_pair(tokens_a, tokens_b, max_length):
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


def pair_up_plus_label(df, mode='train', drop_rate=0.9):
    triplets = []
    #ids = df.id.unique()
    random_drop = np.random.random(size=10000) > drop_rate
    count = 0

    # for each notebook
    for id, df_tmp in tqdm(df.groupby('id')):
        df_tmp_md = df_tmp[df_tmp['cell_type'] == 'markdown']
        df_tmp_code = df_tmp[df_tmp['cell_type'] == 'code']
        df_tmp_code_rank = df_tmp_code['rank'].values
        df_tmp_code_cid = df_tmp_code['cell_id'].values

        for md_cid, rank in df_tmp_md[['cell_id', 'rank']].values:
            # if the code is right after the markdown cell 1, otherwise 0
            labels = np.array([(r == (rank + 1)) for r in df_tmp_code_rank]).astype('int')

            for cd_cid, label in zip(df_tmp_code_cid, labels):
                count += 1
                if label == 1 or mode=='test' or random_drop[count % 10000]:
                    triplets.append([md_cid, cd_cid, label])

    return triplets


def tokenize_and_label(df: pd.DataFrame, source_dict: dict) -> Tuple[np.array, np.array]:
    triplets = pair_up_plus_label(df)

    tokenizer = transformers.RobertaTokenizer.from_pretrained(BASE_MODEL, do_lower_case=True )

    input_ids = np.zeros((len(triplets), SEQ_LEN), dtype="int32")
    attention_mask = np.zeros((len(triplets), SEQ_LEN), dtype="int32")
    segment_ids = np.zeros((len(triplets), SEQ_LEN), dtype="int32")
    labels = np.zeros((len(triplets)), dtype='int32')

    for i, x in enumerate(tqdm(triplets, total=len(triplets))):
        label = x[2]
        markdown_source = source_dict[ x[0]]
        code_source = source_dict[ x[1] ]
        tokens_md = tokenizer.tokenize( markdown_source )[:SEQ_LEN]
        tokens_cd = tokenizer.tokenize( code_source )[:SEQ_LEN]

        # truncate the both tokens together to length less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        truncate_seq_pair( tokens_md, tokens_cd, SEQ_LEN - 3 )
        tokens = ['CLS'] + tokens_md + ['SEP'] + tokens_cd + ['SEP']
        seg_ids = [1] + [0] * (len(tokens_md) + 1) + [1]*(len(tokens_cd) + 1 )

        input = tokenizer.convert_tokens_to_ids(tokens)
        a_mask = [1]*len( input )
        padding_length = SEQ_LEN - len(input)

        # padd 0 to the right
        input +=  ([0] * padding_length)
        a_mask +=  ([0] * padding_length)
        seg_ids += ([0] * padding_length)
        # encoding = tokenizer.encode_plus(
        #     x,
        #     None,
        #     add_special_tokens=True,
        #     max_length=SEQ_LEN,
        #     padding="max_length",
        #     return_token_type_ids=True,
        #     truncation=True,
        # )
        input_ids[i] = input
        attention_mask[i] = a_mask
        segment_ids[i] = seg_ids
        labels[i] = label


    return input_ids, attention_mask, segment_ids, labels

#################################################################################
def get_dataset(
        input_ids: np.array,
        attention_mask: np.array,
        segment_ids: np.array = None,
        labels: Optional[np.array] = None,
        ordered: bool = False,
        repeated: bool = False,
) -> tf.data.Dataset:
    """Return batched and prefetched dataset"""
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices(
            ({"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": segment_ids}, labels)
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            {"input_ids": input_ids, "attention_mask": attention_mask,  "token_type_ids": segment_ids}
        )
    if repeated:
        dataset = dataset.repeat()
    if not ordered:
        dataset = dataset.shuffle(1024)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


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

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


#################################### CELL inference ######################################################################
paths = glob.glob(os.path.join(DATA_PATH, "test", "*.json"))

test_df = pd.concat([read_notebook(x) for x in tqdm(paths, total=len(paths))])
test_df = test_df.rename_axis("cell_id").reset_index()

test_df["rank"] = test_df.groupby(["id", "cell_type"]).cumcount()
test_df["pct_rank"] = test_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)

display(test_df)

input_ids, attention_mask, segment_ids, labels = tokenize_and_label(test_df, dict(zip(test_df['cell_id'].values, test_df['source'].values)))
test_dataset = get_dataset(
    input_ids=input_ids,
    attention_mask=attention_mask,
    segment_ids = segment_ids,
    ordered=True,
)
model = get_model()
model.load_weights("model_0.h5")
y_pred = model.predict(test_dataset)

preds_copy = y_pred

pred_vals = []
count = 0
for id, df_tmp in tqdm(test_df.groupby('id')):
  df_tmp_mark = df_tmp[df_tmp['cell_type']=='markdown']
  df_tmp_code = df_tmp[df_tmp['cell_type']!='markdown']
  df_tmp_code_rank = df_tmp_code['rank'].rank().values
  N_code = len(df_tmp_code_rank)
  N_mark = len(df_tmp_mark)

  preds_tmp = preds_copy[count:count+N_mark * N_code]

  count += N_mark * N_code
    # for each markdown cell,
  for i in range(N_mark):
    pred = preds_tmp[i*N_code:i*N_code+N_code]

    softmax = np.exp((pred-np.mean(pred)) *20)/np.sum(np.exp((pred-np.mean(pred)) *20))

    rank = np.sum(softmax * df_tmp_code_rank)
    pred_vals.append(rank)

######################################### calculate Kendal_tau ##########################################################
test_df.loc[test_df["cell_type"] == "markdown", "pred"] = pred_vals

df_orders = pd.read_csv(os.path.join(DATA_PATH, "train_orders.csv"),   index_col='id').squeeze("columns").str.split()  # Split the string representation of cell_ids into a list

y_dummy = test_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
kendall_tau(df_orders.loc[y_dummy.index], y_dummy)

################################### SUBMIT CELL ###########################################################################
test_df.loc[test_df["cell_type"] == "markdown", "pct_rank"] = y_pred
df = test_df.sort_values("pct_rank").groupby("id", as_index=False)["cell_id"].apply(lambda x: " ".join(x))
df.rename(columns={"cell_id": "cell_order"}, inplace=True)
df.to_csv("submission.csv", index=False)
display(df)