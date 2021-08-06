generate-data:
	mkdir -p data
	python generate_data.py

train:
	mkdir -p experiments
	python main.py --nx 30 --maxiter 1000 --name "eid"

compute-error:
	python compute_error.py --name "eid"

compute-test-error:
	python compute_error.py --test-mode 1 --dataset "test" --name "eid"

plot:
	mkdir -p plots/paper
	python plot_snapshots.py --time 1.0 --name "eid"
	python plot_snapshots.py --time 3.5 --name "eid"

plot-test:
	mkdir -p plots/paper
	python plot_snapshots.py --time 1.0 --base --test --dataset "test" --name "eid"
	python plot_snapshots.py --time 3.5 --base --test --dataset "test" --name "eid"

plot-time:
	mkdir -p plots/paper
	python plot_time.py --dataset "time" --t1 50.0 --name "eid"

plot-velocity:
	mkdir -p plots/paper
	python plot_velocity.py --name "eid"
