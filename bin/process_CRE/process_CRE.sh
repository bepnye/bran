#!/usr/bin/env bash

input_dir=${CDR_IE_ROOT}/data/cre
word_piece_vocab=${input_dir}/word_pieces.txt
processed_dir=${input_dir}/train
proto_dir=${processed_dir}/protos
max_len=50000
# replace infrequent tokens with <UNK>
min_count=5

# convert processed data to tensorflow protobufs
python ${CDR_IE_ROOT}/src/processing/labled_tsv_to_tfrecords.py --text_in_files ${processed_dir}/\*tive_\*CRE\* --out_dir ${proto_dir} --max_len ${max_len} --num_threads 10 --multiple_mentions --tsv_format --min_count ${min_count}

# convert ner data to tf protos
python ${CDR_IE_ROOT}/src/processing/ner_to_tfrecords.py --in_files ${processed_dir}/ner_\* --out_dir ${proto_dir} --load_vocab ${proto_dir} --num_threads 5
