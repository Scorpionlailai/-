from pathlib import Path
import sys
q=Path(__file__).resolve().parent
core=Path('/home/xyw/paper/E24')
sys.path[:0]=[str(core/'src'),str(core),'/home/xyw/paper/E23/scripts',str(q)]
from adapters.run import main
if __name__=='__main__':main()
