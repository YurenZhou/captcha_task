import logging

from src import Captcha

def main(im_path, save_path):

    logging.basicConfig(level=logging.INFO)
    logging.info("----- Demonstration of the Captcha Identifier started -----")

    logging.info("** Test1: Loading trained model from file")
    captcha = Captcha()
    captcha(im_path, save_path)

    logging.info("** Test2: Re-train a model")
    captcha = Captcha(load_model=False)
    captcha(im_path, save_path)


    logging.info("----- Demonstration of the Captcha Identifier ended -----")


if __name__ == "__main__":
    im_path = "./sampleCaptchas/input/input100.jpg"
    save_path = "./results/output100.txt"
    main(im_path, save_path)
