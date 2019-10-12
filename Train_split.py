import pickle
import sys
f = pickle.load(open("s.txt", "rb"))
code = {}
rf = open("processPro_t.txt", "r")
lines = rf.readlines()
for i in range(int(len(lines) / 3)):
    code[int(lines[3 * i])] = (lines[3 * i + 1], lines[3 * i + 2])
for x in f:
	wf = open(x + ".txt", "w")
	for y in f[x]:
	    try:
	    	wf.write(code[y[0]][0])
	    	wf.write(code[y[0]][1])
	        #print(code[x[0]])
	        #print(code[x[2]])
	    except:
	        print sys.exc_info()
	        print(y[0], y[2])
