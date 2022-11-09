CUDA_VISIBLE_DEVICES=1 python3 Evoformer/run_pretrained_openfold.py \
        data/'Metal Ion Binding'/fasta \
        /home/public/OpenFold/dataset/pdb_mmcif/mmcif_files/ \
        --use_precomputed_alignments output/alignments\
        --uniref90_database_path  /home/public/OpenFold/dataset/uniref90/uniref90.fasta \
        --mgnify_database_path  /home/public/OpenFold/dataset/mgnify/mgy_clusters.fa \
        --pdb70_database_path  /home/public/OpenFold/dataset/pdb70/pdb70 \
        --uniclust30_database_path  /home/public/OpenFold/dataset/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
        --output_dir ./output \
        --model_device 'cuda:0' \
        --jackhmmer_binary_path ./lib/conda/envs/plms/bin/jackhmmer \
        --hhblits_binary_path ./lib/conda/envs/plms/bin/hhblits \
        --hhsearch_binary_path ./lib/conda/envs/plms/bin/hhsearch \
        --kalign_binary_path ./lib/conda/envs/plms/bin/kalign \
        --config_preset 'model_1_ptm' \
        --openfold_checkpoint_path /home/public/OpenFold/resources/openfold_params/finetuning_ptm_2.pt
