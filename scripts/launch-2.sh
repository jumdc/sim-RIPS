# python toyxp/train-sim.py \
# ++prefix=vicREG-128 \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=200 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++topological.max_dim=1 \
# ++batch_size=128 \
# ++supervised.batch_size=128 \
# ++logger.mode=offline \
# ++self_supervised.loss=vicreg

# python toyxp/train-sim.py \
# ++prefix=vicREG-64 \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=200 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++topological.max_dim=1 \
# ++batch_size=64 \
# ++supervised.batch_size=64 \
# ++logger.mode=offline \
# ++self_supervised.loss=vicreg

python toyxp/train-sim.py \
++prefix=simRIPS-128 \
++self_supervised.pretrained=True \
++self_supervised.epochs=200 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=128 \
++supervised.batch_size=128 \
++self_supervised.loss=simrips

python toyxp/train-sim.py \
++prefix=simRIPS-64-only-topo \
++self_supervised.pretrained=True \
++self_supervised.epochs=200 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=64 \
++supervised.batch_size=64 \
++topological.w_l2=0 \
++self_supervised.loss=simrips

python toyxp/train-sim.py \
++prefix=fully-supervised-1024 \
++self_supervised.pretrained=False \
++supervised.epochs=200 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++supervised.batch_size=1024