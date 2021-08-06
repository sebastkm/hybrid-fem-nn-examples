generate-data:
	mkdir -p data
	python generate_data.py

train:
	mkdir -p experiments
	python main.py --seed 123 --maxiter-adam 0 --maxiter-bfgs 10000 --feed-obs --name "eid"

plot:
	mkdir -p plots/paper
	python plot.py --name "eid"

plot-2d:
	mkdir -p plots/paper/v
	python plot_2d.py --time 15.0 --save last --name "eid"

generate-test-data:
	mkdir -p data
	python generate_data.py --stimulus 1 --name "test_data" --t1 100.0 --crossed

plot-test:
	mkdir -p plots/paper
	python plot.py --stimulus 1 --dataset "test_data" --t1 100.0 --crossed --name "eid"

plot-test-2d:
	mkdir -p plots/paper/v
	python plot_2d.py --stimulus 1 --dataset "test_data" --t1 100.0 --crossed --time 12.0 --save last --name "eid"
