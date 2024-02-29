python toyxp/train-sim.py \
++prefix=vicREG-64 \
++self_supervised.pretrained=True \
++self_supervised.epochs=200 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++batch_size=64 \
++supervised.batch_size=64 \
++logger.mode=offline \
++self_supervised.loss=vicreg

python toyxp/train-sim.py \
++prefix=vicREG-256 \
++self_supervised.pretrained=True \
++self_supervised.epochs=200 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++batch_size=256 \
++supervised.batch_size=256 \
++logger.mode=offline \
++self_supervised.loss=vicreg

python toyxp/train-sim.py \
++prefix=vicREG-1024 \
++self_supervised.pretrained=True \
++self_supervised.epochs=200 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++batch_size=1024 \
++supervised.batch_size=1024 \
++logger.mode=offline \
++self_supervised.loss=vicreg

python toyxp/train-sim.py \
++prefix=simCLR-256 \
++self_supervised.pretrained=True \
++self_supervised.epochs=200 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=256 \
++supervised.batch_size=256 \
++logger.mode=offline \
++self_supervised.loss=simclr

python toyxp/train-sim.py \
++prefix=simCLR-1024 \
++self_supervised.pretrained=True \
++self_supervised.epochs=200 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++topological.max_dim=1 \
++batch_size=1024 \
++supervised.batch_size=1024 \
++logger.mode=offline \
++self_supervised.loss=simclr