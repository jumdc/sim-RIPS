python toyxp/train-sim.py \
++prefix=fully-supervised \
++self_supervised.pretrained=True \
++self_supervised.epochs=2 \
++supervised.epochs=10 \
++supervised.lr=1e-4 \
++overfit_batches=2 \
++log=True \
++batch_size=128 \
++num_workers=30 \
++log=False \
++self_supervised.loss=vicREG
