python toyxp/train-sim.py \
++prefix=TOPOReg-64-l-0.1 \
++self_supervised.pretrained=True \
++self_supervised.epochs=100 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=64 \
++supervised.batch_size=64 \
++self_supervised.lr=1e-4 \
++topological.w_topo=0.1 \
++logger.mode=offline \
++self_supervised.loss=simrips

python toyxp/train-sim.py \
++prefix=TOPOReg-64 \
++self_supervised.pretrained=True \
++self_supervised.epochs=100 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=64 \
++supervised.batch_size=64 \
++self_supervised.lr=1e-4 \
++logger.mode=offline \
++self_supervised.loss=simrips

python toyxp/train-sim.py \
++prefix=TOPOReg-128 \
++self_supervised.pretrained=True \
++self_supervised.epochs=100 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=128 \
++supervised.batch_size=128 \
++self_supervised.lr=1e-4 \
++logger.mode=offline \
++self_supervised.loss=simrips


python toyxp/train-sim.py \
++prefix=TOPOReg-128-l-0.1 \
++self_supervised.pretrained=True \
++self_supervised.epochs=100 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=128 \
++supervised.batch_size=128 \
++self_supervised.lr=1e-4 \
++topological.w_topo=0.1 \
++logger.mode=offline \
++self_supervised.loss=simrips
