cd ../../ # gets to src

../venv/bin/python3 run_action_gin.py ../configs/lealla.gin olpeat \
    --damuel_tokens=/lnet/troja/work/people/farhan/outputs/pipelines/lealla_at_damuel \
    --mewsli_tokens=/lnet/troja/work/people/farhan/outputs/pipelines/lealla_at_mewsli \
    --workdir=/lnet/troja/work/people/farhan/outputs/olpeat/ \
    --recalls="[1,10,100]" \
    --languages="["ar", "de", "en", "es", "fa", "ja", "sr", "ta", "tr"]"
