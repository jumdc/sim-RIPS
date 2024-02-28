python toyxp/train-sim.py \
++prefix=checks \
++self_supervised.pretrained=True \
++self_supervised.epochs=2 \
++supervised.epochs=3 \
++supervised.lr=1e-4 \
++overfit_batches=2 \
++topological.max_dim=1 \
++batch_size=128 \
++log=False \
++ckpt=True \
++self_supervised.loss=topo