python toyxp/train-sim.py \
++prefix=checks \
++self_supervised.pretrained=True \
++self_supervised.epochs=2 \
++supervised.epochs=3 \
++supervised.lr=1e-4 \
++overfit_batches=2 \
++topological.max_dim=1 \
++batch_size=128 \
++log=True \
++ckpt=False \
++topological.w_topo=0.1 \
++self_supervised.loss=simrips