python src/train-sim.py \
++logger.project=test \
++prefix=checks \
++self_supervised.pretrained=True \
++self_supervised.epochs=15 \
++supervised.epochs=3 \
++supervised.lr=1e-4 \
++overfit_batches=2 \
++self_supervised.dtm_reg=True \
++self_supervised.cov_coef=0 \
++self_supervised.std_coef=0 \
++batch_size=64 \
++log=False \
++ckpt=False \
++self_supervised.loss=vicreg