f = open("processPro_t.txt", "w")
f1 = open("processPro.txt", "r")
f2 = open("processPro_s.txt", "r")
f.write(f1.read() + f2.read())