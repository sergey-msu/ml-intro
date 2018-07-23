import sys
import time as t
import datetime as dt
import utils
#from week1 import header, run
#from week2 import header, run
#from week3 import header, run
#from week4 import header, run
#from week5 import header, run
#from week6 import header, run
from week7 import header, run


def main(args=None):
    args = args or sys.argv[1:]

    utils.PRINT.HEADER(header())
    print('STARTED at ', dt.datetime.now(), 'with args: ', args)
    start = t.time()
    run()
    end = t.time()
    utils.PRINT.HEADER('DONE in {}s ({}m)'.format(round(end-start, 2), round((end-start)/60, 2)))

    return

if __name__=='__main__':
    main(sys.argv[1:])
