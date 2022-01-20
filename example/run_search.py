import sys

sys.path.append('/hpctmp/e0444250/TJU_GongGA/heteroga/heteroga')

from setting import ReadSetandCalc
from heteroga import HeterogaSurface
from calculator import LaspCalculator

lasp_calc = LaspCalculator(
    lasp_command='mpirun -np 8 /home/svu/e0444250/install/LASP_FULL_pro2_0_1_intel12_sss/Src/lasp',
    lasp_in='/hpctmp/e0444250/TJU_GongGA/heteroga/example/lasp.in',
    pot_file='/hpctmp/e0444250/TJU_GongGA/heteroga/example/PtO.pot')

conf = ReadSetandCalc(file='config.yaml', calc=lasp_calc)

HeterogaSurface(conf=conf).run()
