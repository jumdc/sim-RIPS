python toyxp/train-sim.py \
++prefix=simCLR \
++self_supervised.pretrained=True \
++self_supervised.epochs=100 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++batch_size=1024