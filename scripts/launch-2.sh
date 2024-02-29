python toyxp/train-sim.py \
++prefix=simCLR-128 \
++self_supervised.pretrained=True \
++self_supervised.epochs=200 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=128 \
++supervised.batch_size=128 \
++logger.mode=offline \
++self_supervised.loss=simclr