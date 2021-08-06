generate-data:
	mkdir -p data
	python generate_data.py

train:
	mkdir -p experiments
	python main.py -nx 80 --maxiter 1000 --name "eid"

plot:
	mkdir -p plots
	python plot.py --name "eid"
