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
SEQ_LEN = 128
RANDOM_STATE = 42
NO_EPOCHS = 20

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
    BATCH_SIZE = 128
    NB_LIMIT = 10

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

def pair_up_with_label(df, mode='train', drop_rate=0.9):
    triplets = []
    random_drop = np.random.random(size=10000) > drop_rate
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
                if label == 1 or mode=='test' or random_drop[count % 10000]:
                    triplets.append([md_cid, cd_cid, label])

    return triplets


def cleanup_text(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def tokenize_and_label(df: pd.DataFrame, source_dict: dict, mode:str ) -> Tuple[np.array, np.array]:
    triplets = pair_up_with_label(df, mode=mode)

    tokenizer = transformers.RobertaTokenizer.from_pretrained(BASE_MODEL, do_lower_case=True )

    input_ids = np.zeros((len(triplets), SEQ_LEN), dtype="int32")
    attention_mask = np.zeros((len(triplets), SEQ_LEN), dtype="int32")
    segment_ids = np.zeros((len(triplets), SEQ_LEN), dtype="int32")
    labels = np.zeros((len(triplets)), dtype='int32')

    for i, x in enumerate(tqdm(triplets, total=len(triplets))):
        markdown_source = cleanup_text( source_dict[ x[0]] )
        code_source = cleanup_text(source_dict[ x[1] ])
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
        input_ids[i] = input
        attention_mask[i] = a_mask
        segment_ids[i] = seg_ids
        labels[i] = x[2]


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

paths = glob.glob(os.path.join(DATA_PATH, "train", "*.json"))
if NB_LIMIT is not None:
    paths = paths[:NB_LIMIT]

notebooks_train = [ read_notebook(path) for path in tqdm(paths, desc='Train NBs')]

df =  pd.concat(notebooks_train).set_index('id', append=True).swaplevel().sort_index(level='id', sort_remaining=False)

df_orders = pd.read_csv(os.path.join(DATA_PATH, "train_orders.csv"),   index_col='id').squeeze("columns").str.split()  # Split the string representation of cell_ids into a list

df_orders_ = df_orders.to_frame().join(  df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
                                            how='right')

ranks = {}
for id_, cell_order, cell_id in df_orders_.itertuples():
    ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

df_ranks = (
    pd.DataFrame
    .from_dict(ranks, orient='index')
    .rename_axis('id')
    .apply(pd.Series.explode)
    .set_index('cell_id', append=True)
)

df_ancestors = pd.read_csv(
    os.path.join(DATA_PATH, "train_ancestors.csv"),
    usecols=["id", "ancestor_id"],
    index_col="id",
)

df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])
#df = df.dropna()
display(df)

dict_cellid_source = dict(zip(df['cell_id'].values, df['source'].values))

##################################### CELL ##############################################################################
input_ids, attention_mask, segment_ids, labels = tokenize_and_label(df, dict_cellid_source, 'train')

#groups = df["ancestor_id"].to_numpy()

print("input_ids:", input_ids.shape)
print("attention_mask:", attention_mask.shape)
print("segment_ids:", attention_mask.shape)
print("labels:", labels.shape)


#################################### train ############################################################################
input_ids, attention_mask, segment_ids, labels= shuffle(
    input_ids, attention_mask, segment_ids, labels, random_state=RANDOM_STATE
)
kfold = KFold(n_splits=N_SPLITS)

for i, (train_index, val_index) in enumerate(kfold.split(input_ids, labels)):
    if TPU is not None:
        tf.tpu.experimental.initialize_tpu_system(TPU)

    with strategy.scope():
        model = get_model()
        model.summary()

        train_dataset = get_dataset(
            input_ids=input_ids[train_index],
            attention_mask=attention_mask[train_index],
            segment_ids= segment_ids[train_index],
            labels=labels[train_index],
            repeated=False,
        )
        val_dataset = get_dataset(
            input_ids=input_ids[val_index],
            attention_mask=attention_mask[val_index],
            segment_ids=segment_ids[val_index],
            labels=labels[val_index],
            ordered=False,
        )

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            batch_size = BATCH_SIZE,
            epochs=NO_EPOCHS,
            verbose=1,
        )

    model.save_weights(f"model_{i}.h5")
    break



exit()
#################################### CELL inference ######################################################################
paths = glob.glob(os.path.join(DATA_PATH, "test", "*.json"))

df = pd.concat([read_notebook(x) for x in tqdm(paths, total=len(paths))])
df = df.rename_axis("cell_id").reset_index()

df["rank"] = df.groupby(["id", "cell_type"]).cumcount()
df["pct_rank"] = df.groupby(["id", "cell_type"])["rank"].rank(pct=True)

display(df)

input_ids, attention_mask, segment_ids, labels = tokenize_and_label(df, dict(zip(df['cell_id'].values, df['source'].values)))
test_dataset = get_dataset(
    input_ids=input_ids,
    attention_mask=attention_mask,
    segment_ids = segment_ids,
    ordered=True,
)
model = get_model()
model.load_weights("model_0.h5")
y_pred = model.predict(test_dataset)

################################### SUBMIT CELL ###########################################################################
df.loc[df["cell_type"] == "markdown", "pct_rank"] = y_pred
df = df.sort_values("pct_rank").groupby("id", as_index=False)["cell_id"].apply(lambda x: " ".join(x))
df.rename(columns={"cell_id": "cell_order"}, inplace=True)
df.to_csv("submission.csv", index=False)
display(df)