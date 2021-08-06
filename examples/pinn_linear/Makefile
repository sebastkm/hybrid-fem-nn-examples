hybrid: init
	./run_hybrid.sh

pinn: init
	./run_pinn.sh

plot-coeff: init
	python plot_all.py --names hybrid_1 pinn_1 w_pinn_1 fem_l2 fem_h1 --legends "Hybrid" "PINN" "PINN (Adjusted weights)" "Pointwise" "Pointwise (H_0^1 reg.)"

plot-errors: init
	python plot_errors.py --names hybrid_1 w_pinn_1 fem_h1 --legends "Hybrid" "PINN (Adjusted weights)" "Pointwise (H_0^1 reg.)"

fem-l2: init
	python fem.py --maxiter 1000 --reg l2 --name "fem_l2"

fem-h1: init
	python fem.py --maxiter 1000 --reg h1 --name "fem_h1"

generate-data: init
	python generate_data.py

init:
	mkdir -p logs
	mkdir -p experiments
	mkdir -p data
	mkdir -p plots
