python toyxp/train-sim.py \
++prefix=vicREG-200 \
++self_supervised.pretrained=True \
++self_supervised.epochs=200 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++batch_size=1024 \
++num_workers=20 \
++logger.mode=offline \
++self_supervised.loss=vicreg