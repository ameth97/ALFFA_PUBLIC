#!/bin/bash

# initialization PATH
. ./kaldi-scripts/00_init_paths.sh || { echo -e "\n00_init_paths.sh expected.\n"; exit; }


##### DATA PREPARAT
##### ASR BUILDING #####
# initialization commands
. ./cmd.sh || { echo -e "\n cmd.sh expected.\n"; exit; }



 ## DNN
echo
echo "===== DNN DATA PREPARATION ====="
echo
 # Config:
gmmdir=exp/system1/tri3b
data_fmllr=system1
stage=0 # resume training with --stage=N
# # End of config.
. utils/parse_options.sh || exit 1;
# #

 if [ $stage -le 0 ]; then
   # Store fMLLR features, so we can train on them easily,
   # dev
   dir=$data_fmllr/dev
   steps/nnet/make_fmllr_feats.sh --nj 2 --cmd run.pl \
      --transform-dir $gmmdir/decode_dev \
      $dir data/dev $gmmdir $dir/log $dir/data || exit 1
   # test
   dir=$data_fmllr/test
   steps/nnet/make_fmllr_feats.sh --nj 2 --cmd run.pl \
      --transform-dir $gmmdir/decode_test \
      $dir data/test $gmmdir $dir/log $dir/data || exit 1
   # train
   dir=$data_fmllr/train
   steps/nnet/make_fmllr_feats.sh --nj 14 --cmd run.pl \
      --transform-dir ${gmmdir}_ali \
      $dir data/train $gmmdir $dir/log $dir/data || exit 1
   # split the data : 90% train 10% cross-validation (held-out)
   utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

echo
 echo "===== DNN DATA TRAINING ====="
 echo
$cuda_cmd=run.pl
echo $cuda_cmd
 # Training
 if [ $stage -le 1 ]; then
   # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
   dir=exp/system1/dnn4b_pretrain-dbn
   (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null) & # forward log
   $cuda_cmd $dir/log/pretrain_dbn.log \
      steps/nnet/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 14 $data_fmllr/train $dir || exit 1;
 fi

 if [ $stage -le 2 ]; then
   # Train the DNN optimizing per-frame cross-entropy.
   dir=exp/system1/dnn4b_pretrain-dbn_dnn
   ali=${gmmdir}_ali
   feature_transform=exp/system1/dnn4b_pretrain-dbn/final.feature_transform
   dbn=exp/system1/dnn4b_pretrain-dbn/2.dbn
 (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null) & # forward log
   # Train
   $cuda_cmd $dir/log/train_nnet.log \
   $dir/log/train_nnet.log \ 
     steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
     $data_fmllr/${TRAIN_DIR}_tr90 $data_fmllr/${TRAIN_DIR}_cv10 lang $ali $ali $dir || exit 1;
   # Decode (reuse HCLG graph)
   steps/nnet/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
     $gmmdir/graph $data_fmllr/dev $dir/decode_dev || exit 1;
   steps/nnet/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
     $gmmdir/graph $data_fmllr/test $dir/decode_test || exit 1;
 fi

 # Sequence training using sMBR criterion, we do Stochastic-GD 
 # with per-utterance updates. We use usually good acwt 0.1
 dir=exp/system1/dnn4b_pretrain-dbn_dnn_smbr
 srcdir=exp/system1/dnn4b_pretrain-dbn_dnn
 acwt=0.1

 if [ $stage -le 3 ]; then
   # First we generate lattices and alignments:
   steps/nnet/align.sh --nj 14 --cmd "$train_cmd" \
     $data_fmllr/train lang $srcdir ${srcdir}_ali || exit 1;
   steps/nnet/make_denlats.sh --nj 14 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
     $data_fmllr/train lang $srcdir ${srcdir}_denlats || exit 1;
 fi

 if [ $stage -le 4 ]; then
   # Re-train the DNN by 2 iterations of sMBR 
   steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
     $data_fmllr/train lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
   # Decode
   for ITER in 1 2 3 4 5 6; do
     steps/nnet/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config \
       --nnet $dir/${ITER}.nnet --acwt $acwt \
       $gmmdir/graph $data_fmllr/dev $dir/decode_dev_it${ITER} || exit 1;
     steps/nnet/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config \
       --nnet $dir/${ITER}.nnet --acwt $acwt \
       $gmmdir/graph $data_fmllr/test $dir/decode_test_it${ITER} || exit 1;
   done 
 fi

 echo Success
 exit 0


# Getting results [see RESULTS file]
for x in exp/system1/*/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done > exp/system1/RESULTS

echo
echo "===== See results in 'exp/system1/RESULTS' ====="
echo

