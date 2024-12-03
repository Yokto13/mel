#!/bin/bash

# An example on how to build the multilingual dataset

python run_action_gin.py ../configs/general.gin create_multilingual_dataset --source_dir=/lnet/work/home-students-external/farhan/troja/outputs/finetuning_damuel_2 --langs=[af,be,ca,da,el,es,eu,fi,ga,gl,hi,hu,id,ja,la,lv,mt,nn,pt,ru,sk,sr,ta,tr,uk,vi,zh,ar,bg,cs,de,en,et,fa,fr,gd,he,hr,hy,it,ko,lt,mr,nl,pl,ro,se,sl,sv,te,ug,ur,wo] --dest_dir=/lnet/work/home-students-external/farhan/troja/outputs/all2/