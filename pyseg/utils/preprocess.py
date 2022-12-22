import os

def exp_generator():
    if not os.path.exists("exp"):
        os.makedirs("exp")

    exp = os.listdir("exp")
    exp = [x for x in exp if not x.startswith(".")]
    
    if not exp:
        os.mkdir("exp/0")
        last = 0
    else:
        last = list(map(int, exp))
        last.sort()
        last = last[-1]
        last+= 1
        
        os.mkdir(f"exp/{last}")
        
    return last

if __name__=="__main__":
    exp_generator()