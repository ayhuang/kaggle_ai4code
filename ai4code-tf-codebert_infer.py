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


def chain_blocks( block, t, cells_to_sort, buddies, blocks, head_index):
    if t in blocks.keys():
        block.extend( blocks[t])
        blocks.pop(t, None)
    elif t in cells_to_sort:
        t1 = buddies[t]
        block.append( t1 )
        if t > head_index:
            cells_to_sort.remove(t) # remove the list to avoid duplicate
        blocks.pop(t, None)
        chain_blocks( block, t1, cells_to_sort, buddies, blocks, head_index)

    return

# this will sort an array based on relative pair wise order
# assuming only the bottom portion of the original list needs to be sorted
# the relative rank is given as one-hot-index of the after index, i.e  1 means the index is after the index being sorted 0 means otherwise
def sort_with_pairwise_rank( orig_order, relative_rank):
    no_elements = len(orig_order)
    no_to_sort = int(len( relative_rank)/no_elements)

    fix_elements = no_elements - no_to_sort
    cells_to_sort = [i for i in range(fix_elements, no_elements)]

    trailling_buddy= [None]*no_elements
    # initialize the fixed elements, if one of these is found to be a buddy, then it will be removed.
    blocks = dict((x,[x]) for x in range(0, fix_elements))
    #try to pair up the cells
    min_sum = 1000.0
    last_cell = -1
    for i in range(no_to_sort):
        tmp_rel_rank = relative_rank[i * no_elements:(i + 1) * no_elements]
        # it is possible a markdown cell may be the last cell in the NB, it then will not have a trailing budding
        # however, there doesn't seem to be a deterministic way to find that out. So we will just have to make some
        # arbitrary assumption here.
        sum_0 = np.linalg.norm(tmp_rel_rank)
        buddy = np.argmax(tmp_rel_rank)
        trailling_buddy[i + fix_elements] = buddy
        if buddy < fix_elements and sum_0 > 1.0e-9:
            blocks.pop(buddy, None)

        if sum_0 < min_sum:
            min_sum = sum_0
            last_cell = i+fix_elements

    if last_cell > 0 and min_sum < 0.4:
        trailling_buddy[last_cell] = 5000; # cell without a buddy, must be the last cell.


    # combine to form contigious blocks  i.e., we  may have 12->[17] and 17->[10], in which case,
    # [12,17,10] forms a single block
    for k in cells_to_sort:
        t= trailling_buddy[k]
        block=[k, t]
        chain_blocks( block, t, cells_to_sort, trailling_buddy, blocks, k)
        blocks[k] = block

    #blocks may be like [[4,7,11,10],[5,3],[6,1],[8,9,2]]
    lst = list(blocks.values())
    lst.sort( key=lambda x:x[-1])
    #flatten the blocks into a list
    new_order =  [x for block in lst for x in block]
    return [new_order.index(x) for x in range(no_elements)]





#################################### CELL inference ######################################################################
paths = glob.glob(os.path.join(DATA_PATH, "val", "*.json"))

test_df = pd.concat([read_notebook(x) for x in tqdm(paths, total=len(paths))])
#test_df = test_df.rename_axis("cell_id").reset_index()

df_orders = pd.read_csv(os.path.join(DATA_PATH, "train_orders.csv"),   index_col='id').squeeze("columns").str.split()  # Split the string representation of cell_ids into a list

df_orders_ = df_orders.to_frame().join(  test_df.reset_index('cell_id').groupby('id')['cell_id'].apply(list), how='right')

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
val_df = test_df.reset_index().merge(df_ranks, on=["id", "cell_id"])#.merge(df_ancestors, on=["id"])

test_df["rank"] = test_df.groupby(["id"]).cumcount()
test_df["pred"] = test_df.groupby(["id"])["rank"].rank(pct=False)
#val_df["pct_rank"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)

display(val_df)

input_ids, attention_mask, segment_ids, labels = tokenize_and_label(val_df, dict(zip(val_df['cell_id'].values, val_df['source'].values)), mode='test')
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

count = 0
for id, df_tmp in tqdm(test_df.groupby('id')):
    print( f'processing {id}')
    df_tmp_md = df_tmp[df_tmp['cell_type']=='markdown']

     #original order of all cells, markdown and code in a NB
    orig_order = df_tmp['pred'].values
    no_cell = len(orig_order)
    no_md = len(df_tmp_md)

    preds_tmp = preds_copy[count:count+no_md * no_cell]

    count += no_md * no_cell
    new_order = sort_with_pairwise_rank(orig_order, preds_tmp )
    test_df.loc[test_df['id'] == id, 'pred'] = new_order

######################################### calculate Kendal_tau ##########################################################
y_dummy = test_df.reset_index().sort_values('pred').groupby('id')['cell_id'].apply(list)
k = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
print(f"kendall_tau = {k}")
