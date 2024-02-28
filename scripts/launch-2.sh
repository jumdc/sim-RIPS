# python toyxp/train-sim.py \
# ++prefix=simRIPS \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=100 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++topological.max_dim=1 \
# ++batch_size=512 \
# ++log=True \
# ++self_supervised.loss=topo

python toyxp/train-sim.py \
++prefix=fully-supervised-128 \
++self_supervised.pretrained=False \
++supervised.epochs=200 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++supervised.batch_size=128

python toyxp/train-sim.py \
++prefix=fully-supervised-256 \
++self_supervised.pretrained=False \
++supervised.epochs=200 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++supervised.batch_size=256

python toyxp/train-sim.py \
++prefix=simCLR-128 \
++self_supervised.pretrained=True \
++self_supervised.epochs=200 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=64 \
++supervised.batch_size=128 \
++self_supervised.loss=simclr

python toyxp/train-sim.py \
++prefix=vicREG-128 \
++self_supervised.pretrained=True \
++self_supervised.epochs=200 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=128 \
++supervised.batch_size=128 \
++self_supervised.loss=vicreg