CUDA_VISIBLE_DEVICES=1 python3 Evoformer/run_pretrained_openfold.py \
        /home/public/OpenFold/fasta_dir \
        /home/public/OpenFold/dataset/pdb_mmcif/mmcif_files/ \
        --uniref90_database_path  /home/public/OpenFold/dataset/uniref90/uniref90.fasta \
        --mgnify_database_path  /home/public/OpenFold/dataset/mgnify/mgy_clusters.fa \
        --pdb70_database_path  /home/public/OpenFold/dataset/pdb70/pdb70 \
        --uniclust30_database_path  /home/public/OpenFold/dataset/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
        --output_dir /root/openfold/output \
        --bfd_database_path  /home/public/OpenFold/dataset/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
        --model_device 'cuda:0' \
        --jackhmmer_binary_path /opt/anaconda3/envs/openfold_env/bin/jackhmmer \
        --hhblits_binary_path /opt/anaconda3/envs/openfold_env/bin/hhblits \
        --hhsearch_binary_path /opt/anaconda3/envs/openfold_env/bin/hhsearch \
        --kalign_binary_path /opt/anaconda3/envs/openfold_env/bin/kalign \
        --config_preset 'model_1_ptm' \
        --openfold_checkpoint_path /home/public/OpenFold/resources/openfold_params/finetuning_ptm_2.pt
