python src/train-sim.py \
++logger.project="ssl-xplore" \
++prefix=DTM-64-k-10 \
++self_supervised.pretrained=True \
++self_supervised.epochs=20 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++batch_size=64 \
++supervised.batch_size=64 \
++logger.mode=offline \
++self_supervised.lr=1e-4 \
++dtm.k=10 \
++self_supervised.loss=dtm

python src/train-sim.py \
++logger.project="ssl-xplore" \
++prefix=DTM-128 \
++self_supervised.pretrained=True \
++self_supervised.epochs=20 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++self_supervised.dtm_reg=True \
++batch_size=128 \
++supervised.batch_size=128 \
++logger.mode=offline \
++self_supervised.lr=1e-4 \
++dtm.k=10 \
++self_supervised.loss=dtm

python src/train-sim.py \
++logger.project="ssl-xplore" \
++prefix=DTM-256 \
++self_supervised.pretrained=True \
++self_supervised.epochs=20 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++self_supervised.dtm_reg=True \
++batch_size=256 \
++supervised.batch_size=256 \
++logger.mode=offline \
++self_supervised.lr=1e-4 \
++dtm.k=10 \
++self_supervised.loss=dtm


# python src/train-sim.py \
# ++logger.project="ssl-xplore" \
# ++prefix=TOPO-checks-max-dim-1 \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=20 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++topological.max_dim=1 \
# ++topological.w_l2=1 \
# ++topological.w_topo=1 \
# ++batch_size=64 \
# ++supervised.batch_size=64 \
# ++self_supervised.lr=1e-4 \
# ++logger.mode=offline \
# ++self_supervised.loss=simrips

# python src/train-sim.py \
# ++logger.project="ssl-xplore" \
# ++prefix=DTM-reg-256 \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=20 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++self_supervised.dtm_reg=True \
# ++batch_size=128 \
# ++supervised.batch_size=256 \
# ++logger.mode=offline \
# ++self_supervised.lr=1e-4 \
# ++self_supervised.loss=vicreg

# python toyxp/train-sim.py \
# ++prefix=TOPO-reg \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=20 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++topological.max_dim=2 \
# ++batch_size=64 \
# ++supervised.batch_size=64 \
# ++self_supervised.lr=1e-4 \
# ++topological.max_dim=0 \
# ++topological.w_l2=1 \
# ++topological.w_topo=1 \
# ++self_supervised.loss=simrips

# ++logger.mode=offline \



# python toyxp/train-sim.py \
# ++prefix=TOPOReg-64 \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=100 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++topological.max_dim=1 \
# ++batch_size=64 \
# ++supervised.batch_size=64 \
# ++self_supervised.lr=1e-4 \
# ++logger.mode=offline \
# ++self_supervised.loss=simrips

# python toyxp/train-sim.py \
# ++prefix=TOPOReg-128 \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=100 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++topological.max_dim=1 \
# ++batch_size=128 \
# ++supervised.batch_size=128 \
# ++self_supervised.lr=1e-4 \
# ++logger.mode=offline \
# ++self_supervised.loss=simrips


# python toyxp/train-sim.py \
# ++prefix=TOPOReg-128-l-0.1 \
# ++self_supervised.pretrained=True \
# ++self_supervised.epochs=100 \
# ++supervised.epochs=100 \
# ++supervised.lr=1e-4 \
# ++overfit_batches=0 \
# ++log=True \
# ++topological.max_dim=1 \
# ++batch_size=128 \
# ++supervised.batch_size=128 \
# ++self_supervised.lr=1e-4 \
# ++topological.w_topo=0.1 \
# ++logger.mode=offline \
# ++self_supervised.loss=simrips
