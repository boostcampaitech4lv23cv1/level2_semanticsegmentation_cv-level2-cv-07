import os

def exp_generator():
    if not os.path.exists("exp"):
        os.makedirs("exp/")

    exp = os.listdir("exp")
    exp = [x for x in exp if not x.startswith(".")]
    
    if not exp:
        last = 0
    else:
        last = list(map(int, exp))
        last.sort()
        last = last[-1]
        last+= 1
        
    os.makedirs(f"exp/{last}/tensorboard")
    os.makedirs(f"exp/{last}/wandb")
        
    return last

if __name__=="__main__":
    exp_generator()