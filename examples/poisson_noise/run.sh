python main.py --nx 80 --seed 3 --snr 10.0 --layers 2 30 1 --maxiter 1000 --name "reg" --reg 1e-4 > logs/reg.txt 2>&1 &
python main.py --nx 80 --seed 3 --snr 10.0 --layers 2 30 1 --maxiter 10000 --name "no_reg" --reg 0.0 > logs/no_reg.txt 2>&1 &
python main.py --nx 80 --seed 3 --snr 10.0 --layers 2 30 1 --maxiter 10000 --name "reg_long" --reg 1e-4 > logs/reg_long.txt 2>&1 &
