python toyxp/train-sim.py \
++logger.project=test \
++prefix=checks \
++self_supervised.pretrained=True \
++self_supervised.epochs=15 \
++supervised.epochs=3 \
++supervised.lr=1e-4 \
++overfit_batches=2 \
++topological.max_dim=1 \
++self_supervised.cov_coef=0 \
++self_supervised.std_coef=0 \
++batch_size=64 \
++log=False \
++ckpt=False \
++topological.w_topo=0.01 \
++self_supervised.loss=vicreg