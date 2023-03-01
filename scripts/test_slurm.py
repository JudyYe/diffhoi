import time
import hydra 

@hydra.main(config_path="../configs", config_name="volsdf_nogt")
def test(args):
    while True:
        print(time.time())
        time.sleep(100)

if __name__ == "__main__":
    test()