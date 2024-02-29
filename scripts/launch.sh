python toyxp/train-sim.py \
++prefix=simRIPS-64 \
++self_supervised.pretrained=True \
++self_supervised.epochs=100 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=64 \
++supervised.batch_size=64 \
++self_supervised.loss=simrips

# python toyxp/train-sim.py \
# ++prefix=simCLR-64 \
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
# ++self_supervised.loss=simclr

# python toyxp/train-sim.py \
# ++prefix=simCLR-128 \
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
# ++self_supervised.loss=simclr

# python toyxp/train-sim.py \
# ++prefix=simRIPS-128-only-topo \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=200 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++topological.max_dim=1 \
# ++batch_size=128 \
# ++supervised.batch_size=128 \
# ++topological.w_l2=0 \
# ++self_supervised.loss=simrips

# python toyxp/train-sim.py \
# ++prefix=simCLR-512 \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=200 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++topological.max_dim=1 \
# ++batch_size=512 \
# ++supervised.batch_size=512 \
# ++logger.mode=offline \
# ++self_supervised.loss=simclr

# python toyxp/train-sim.py \
# ++prefix=vicREG-512 \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=200 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++topological.max_dim=1 \
# ++batch_size=512 \
# ++supervised.batch_size=512 \
# ++logger.mode=offline \
# ++self_supervised.loss=vicreg

# python toyxp/train-sim.py \
# ++prefix=simRIPS-512 \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=200 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++topological.max_dim=1 \
# ++batch_size=512 \
# ++supervised.batch_size=512 \
# ++logger.mode=offline \
# ++self_supervised.loss=simrips