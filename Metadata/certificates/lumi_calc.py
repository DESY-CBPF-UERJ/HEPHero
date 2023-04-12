
in_file = open("lumi_16_HIPM.txt", "r")

lumi_total = 0
for line in in_file:
    info = line.split('|')
    lumi_total += float(info[6])
    print(info[6])

print("Total: " + str(lumi_total))
in_file.close()
