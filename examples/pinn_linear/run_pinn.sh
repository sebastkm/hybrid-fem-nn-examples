python pinn.py --seed 1 --maxiter 100000 --weights 1. 1. 1. 1. --name pinn_1 > logs/pinn_1.txt 2>&1 &
python pinn.py --seed 2 --maxiter 100000 --weights 1. 1. 1. 1. --name pinn_2 > logs/pinn_2.txt 2>&1 &
python pinn.py --seed 3 --maxiter 100000 --weights 1. 1. 1. 1. --name pinn_3 > logs/pinn_3.txt 2>&1 &
python pinn.py --seed 4 --maxiter 100000 --weights 1. 1. 1. 1. --name pinn_4 > logs/pinn_4.txt 2>&1 &
python pinn.py --seed 5 --maxiter 100000 --weights 1. 1. 1. 1. --name pinn_5 > logs/pinn_5.txt 2>&1 &

python pinn.py --seed 1 --maxiter 10000 --weights 1. 10. 1e+2 1e+3 --name w_pinn_1 > logs/w_pinn_1.txt 2>&1 &
python pinn.py --seed 2 --maxiter 10000 --weights 1. 10. 1e+2 1e+3 --name w_pinn_2 > logs/w_pinn_2.txt 2>&1 &
python pinn.py --seed 3 --maxiter 10000 --weights 1. 10. 1e+2 1e+3 --name w_pinn_3 > logs/w_pinn_3.txt 2>&1 &
python pinn.py --seed 4 --maxiter 10000 --weights 1. 10. 1e+2 1e+3 --name w_pinn_4 > logs/w_pinn_4.txt 2>&1 &
python pinn.py --seed 5 --maxiter 10000 --weights 1. 10. 1e+2 1e+3 --name w_pinn_5 > logs/w_pinn_5.txt 2>&1 &
