python toyxp/train-sim.py \
++prefix=simRIPS \
++self_supervised.pretrained=True \
++self_supervised.epochs=100 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=512 \
++log=True \
++self_supervised.loss=topo

python toyxp/train-sim.py \
++prefix=simRIPS \
++self_supervised.pretrained=True \
++self_supervised.epochs=50 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=512 \
++log=True \
++self_supervised.loss=topo \
++logger.mode=offline





