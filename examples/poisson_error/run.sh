python main.py -nx 5 --seed 2 --maxiter 50000 --layers 2 30 1 --name "nx5" > logs/nx5.txt 2>&1 &
python main.py -nx 10 --seed 2 --maxiter 50000 --layers 2 30 1 --name "nx10" > logs/nx10.txt 2>&1 &
python main.py -nx 20 --seed 2 --maxiter 50000 --layers 2 30 1 --name "nx20" > logs/nx20.txt 2>&1 &
python main.py -nx 40 --seed 2 --maxiter 50000 --layers 2 30 1 --name "nx40" > logs/nx40.txt 2>&1 &
python main.py -nx 80 --seed 2 --maxiter 50000 --layers 2 30 1 --name "nx80" > logs/nx80.txt 2>&1 &


