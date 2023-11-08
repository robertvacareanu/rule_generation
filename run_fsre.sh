#!/bin/bash

for RULE_TYPE in 'surface'
do
    for DATASET in "TACRED" "NYT29"
    do
        python main.py --min_span_length 1 --max_span_length 15 --num_matches 3 --rule_type ${RULE_TYPE} --save_path "fsre_dataset_rules/${DATASET}/${RULE_TYPE}_wordslemmasandtags.jsonl" \
            --fields_word_weight 1 --fields_lemma_weight 1 --fields_tag_weight 1 \
            --docs_dir "/data/nlp/corpora/fs-re-dataset-paper/models/softrules/odinson/${DATASET,,}/docs/" \
            --data_paths \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/few_shot_data/_train_data.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/few_shot_data/_dev_data.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/few_shot_data/_test_data.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/train_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/train_episodes/5_way_1_shots_10K_episodes_3q_seed_160291.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/train_episodes/5_way_1_shots_10K_episodes_3q_seed_160292.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/train_episodes/5_way_1_shots_10K_episodes_3q_seed_160293.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/train_episodes/5_way_1_shots_10K_episodes_3q_seed_160294.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/train_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/train_episodes/5_way_5_shots_10K_episodes_3q_seed_160291.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/train_episodes/5_way_5_shots_10K_episodes_3q_seed_160292.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/train_episodes/5_way_5_shots_10K_episodes_3q_seed_160293.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/train_episodes/5_way_5_shots_10K_episodes_3q_seed_160294.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160291.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160292.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160293.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160294.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160291.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160292.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160293.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160294.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160291.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160292.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160293.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160294.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160291.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160292.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160293.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/${DATASET}/episodes/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160294.json" &
    done
    wait
done
