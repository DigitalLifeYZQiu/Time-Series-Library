export CUDA_VISIBLE_DEVICES=4
#gpu=0
#ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full/04012025_193158/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full-epoch=14.ckpt"
#ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p4_n512_weight_p50_wo_revin_full/07012025_015048/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p4_n512_weight_p50_wo_revin_full-epoch=11.ckpt"
ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long/20012025_231911/ckpt/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long-epoch=01.ckpt"
for seq_len in 40
do
for mask_rate in 0
do
for lr in 0.0005
do
for anomaly_ratio in 0.5
do
for stride in 40
do
echo mask rate $mask_rate
#export CUDA_VISIBLE_DEVICES=$gpu
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --ckpt_path $ckpt_path \
  --root_path ../DATA/SMAP \
  --model_id SMAP \
  --model TimesBERT \
  --data SMAP \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --batch_size 64 \
  --patch_len 4 \
  --enc_in 25 \
  --train_epochs 20 \
  --stride $stride \
  --anomaly_ratio $anomaly_ratio \
  --learning_rate $lr \
  --lradj cosine \
  --ad_mask_type random \
  --mask_rate $mask_rate \
  --date_record
done
done
done
done
done

# > scriptsTimeBert/sota/TimeBert_SMAP.log && bash scriptsTimeBert/sota/TimesBERT_SMAP.sh 2>&1 | tee -a scriptsTimeBert/sota/TimeBert_SMAP.log