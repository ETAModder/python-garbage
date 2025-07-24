import random
import string
def brute_text(target=""):
    alphabet = list(string.ascii_lowercase + string.ascii_uppercase + " " + "'" + ":" + "." + "," + ";" + "\n" + "[" + "]" + "!" + "?" + "-" + "1234567890" + "!@#$%^&*()" + "/<>[]\\|_=+`~")
    result = ""
    i = 0
    
    while result != target:
        letter = random.choice(alphabet)
        guess = result + letter
        print(guess, end="\r", flush=True)
        if letter == target[i]:
            result += letter
            i += 1
            alphabet = list(string.ascii_lowercase + string.ascii_uppercase + " " + "'" + ":" + "." + "," + ";" + "\n" + "[" + "]" + "!" + "?" + "-" + "1234567890" + "!@#$%^&*()" + "/<>[]\\|_=+`~")
brute_text(target="a5/x_&di@V^ML7&0*,?#NR0<Ba5/x_&di@V^ML7&0*,?#NB*8ya5/x_&di@V^ML7&0*,?#NR0<Ba5/x_&di@V^ML7&0*,?#NB*8yF]u~>^m)E8DOj9VqDy/NDi@V^ML7&0*,?#NR0<Ba5/x_&di@V^ML7&0*,?#NR0<BB*8yF]u~>^m)E8DOj9VqDy/ND")