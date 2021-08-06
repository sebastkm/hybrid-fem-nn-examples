fem: init
	python fem.py --maxiter 1000 --reg h1 --name fem

hybrid: init
	./run_hybrid.sh

pinn: init
	./run_pinn.sh

plot-coeffs: init
	python plot_all.py --names hybrid_1 w_pinn_1 fem --legends Hybrid "PINN (Adjusted weights)" "Pointwise (H_0^1 reg.)"

plot-states: init
	python plot_all_states.py --names hybrid_1 w_pinn_1 fem --legends Hybrid "PINN (Adjusted weights)" "Pointwise (H_0^1 reg.)"

plot-loss: init
	python plot_loss.py --names hybrid_1 w_pinn_1 --legends Hybrid "PINN PDE" "PINN Data"

generate-data: init
	python generate_data.py

init:
	mkdir -p logs
	mkdir -p experiments
	mkdir -p data
	mkdir -p plots
