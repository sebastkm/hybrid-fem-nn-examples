train:
	mkdir -p experiments
	mkdir -p logs
	./run.sh

plot:
	mkdir -p plots
	python plot_paper.py --name "reg_1_h10"

plot-loss:
	mkdir -p plots
	python plot_loss.py --names "no_reg_1_long" "reg_1_h10_long" --legends "$$\alpha = 0$$" "$$\alpha = 10^{-4}$$"
