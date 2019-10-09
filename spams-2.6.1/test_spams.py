from __future__ import absolute_import, division, print_function
import six.moves

import sys
import time
import test_utils

all_modules = ['linalg', 'decomp', 'prox','dictLearn']

modules = []

# for debug
simul = False

def usage(lst):
    print("Usage : %s [-32] [test-or-group-name]+" %sys.argv[0])
    print('  Run specified test or group of tests (all by default)')
    print('    -32 : use float32 instead of float64')
    print('  Available groups and tests are:')
    print('%s ' '\n'.join(lst))
    print('\nExamples:')
    print('%s linalg' %sys.argv[0])
    print('%s sort calcAAt' %sys.argv[0])
    sys.exit(1)

def run_test(testname,prog):
    print("** Running %s" %testname)
    if simul:
        return
    err = prog()
    if err != None:
        print("  ERR = %f" %err)

def main(argv):
    tic = time.time()
    lst = []
    is_ok = True
    for s in argv:
        if s[0] == '-':
            if s == '-32':
                test_utils.set_float32()
            else:
                is_ok = False
            continue
        lst.append(s)
    for s in all_modules:
        try:
            exec('import test_%s' %s)
            modules.append(s)
        except:
            print("Removing %s" %s)
    if not is_ok:
        l = []
        for m in modules:
            l.append("%s :" %m)
            # exec('lstm = test_%s.tests' %m)
            lstm = locals()['test_%s' %m].tests
            l.append('  %s' %(' '.join([ lstm[i] for i in range(0,len(lstm),2)])))
        usage(l)
    if(len(lst) == 0):
        lst = modules
    for testname in lst:
        if testname in modules:
            print("**** %s ****" %testname)
            # exec('lstm = test_%s.tests' %testname)
            lstm = locals()['test_%s' %testname].tests
            for i in six.moves.xrange(0,len(lstm),2):
                run_test(lstm[i],lstm[i+1])
            continue
        else:
            found = False
            for m in modules:
                # exec('lstm = test_%s.tests' %m)
                lstm = locals()['test_%s' %m].tests
                for i in six.moves.xrange(0,len(lstm),2):
                    if (lstm[i] == testname):
                        found = True
                        run_test(lstm[i],lstm[i+1])
                        break
                if found:
                    break

            if(not found):
                print("Test %s not found!" %testname)

    tac = time.time()
    print('\nTotal time : %.3fs'  %(tac - tic))

if __name__ == "__main__":
    main(sys.argv[1:])
