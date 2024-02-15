python toyxp/train-sim.py \
++prefix=fully-supervised \
++self_supervised.pretrained=False \
++self_supervised.epochs=2 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++batch_size=128 \
++num_workers=30
