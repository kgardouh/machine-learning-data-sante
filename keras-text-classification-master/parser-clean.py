import re

def find_str(s, char):
    index = 0

    if char in s:
        c = char[0]
        for ch in s:
            if ch == c:
                if s[index:index+len(char)] == char:
                    return index

            index += 1

    return -1

file = open("demofile.txt", "w")

with open('data-sante2.txt', 'r') as f: #open the file
    contents = f.readlines() #put the lines to a variable (list).
    #print(contents)

    #contents = map(lambda s: s.strip(), contents)
    for x in contents:
        file.write((x.split('.'))[0]+','+(x.split('.'))[1].strip())
        file.write('\n')
   
    	
  