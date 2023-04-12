
in_file = open("Summer19UL16APV_V7_MC_Uncertainty_AK4PFchs.txt", "r")
out_up = open("JES_UNC_UP.txt", "w")
out_down = open("JES_UNC_DOWN.txt", "w")

pt_ranges = []
eta_ranges = []

control = 0
N = 3
for line in in_file:
    if control == 1:
        info = line.split(' ')
        for i in range(len(info)):
            if( (i >= 3) and (info[i] != '\n') ):
                if (i-3)%N == 0:
                    pt_ranges.append(float(info[i]))
    if control >= 1:
        info = line.split(' ')
        eta_ranges.append(float(info[0]))
        last_line = line
        
        for i in range(len(info)):
            if( (i >= 4) and (info[i] != '\n') ):
                if (i-4)%N == 0:
                    out_up.write(str(info[i])+"\t")
                if (i-4)%N == 1:
                    out_down.write(str(info[i])+"\t")
        out_up.write("\n")
        out_down.write("\n")
        control += 1
    else:
        control = 1

info = last_line.split(' ')
eta_ranges.append(float(info[1]))

print("pt_ranges:")
print(pt_ranges)
print(len(pt_ranges))
print(" ")
print("eta_ranges:")
print(eta_ranges)
print(len(eta_ranges))

in_file.close()
out_up.close()
out_down.close()
