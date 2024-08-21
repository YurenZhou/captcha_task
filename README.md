##  Captcha Identifier

### Assumptions

1. Assume the unseen captchas to identify are similar to the provided ones:
    - the number of characters remains the same each time
    - the font and spacing is the same each time
    - the background and foreground colors and texture, remain largely the same
    - there is no skew in the structure of the characters
    - the captcha generator, creates strictly 5-character captchas, and each of the characters is either an upper-case character (A-Z) or a numeral (0-9)

### Explanation of the solution

1. Each captcha image is segmented into 5 character images first, each of which contains exactly 1 character
2. Train a CNN model to inter character from the character image:
    1. Since the task is simple and training data is limited, the structure of the CNN is very simple
    2. Optuna is used to tune some of the hyper-parameters with leave-one-out cross-validation. Number of trials is set to be small for quicker run time
    3. After determining the best hyper-parameters, the final model is trained and saved
3. To identifer a unseen captcha:
    1. Segment the captcha into character images
    2. Use the trained model to infer all the characters
    3. Concatenate the characters and output the file

### Basics of the code

1. The code is written in Python 3.10
2. Required Python packages are in requirements.txt and Dockerfile
3. The input data path is currently fixed as ./sampleCaptchas
4. The tested captcha is ./sampleCaptchas/input/input100.jpg
5. The resulted txt file will be put inside ./results
6. Entry of the program is main.py:
    1. One can change the tested captcha and output path inside main.py
7. For the purpose of demonstration:
    1. The tested captcha is identified first using the pre-trained model
    2. Then the model will be re-trained and the tested captcha is identified again using the newly trained model

### Instructions on running the code

#### Method 1: Use Docker (Recommended)

Step 1: Change pwd to the directory of this README.md

Step 2: Build the docker image

```shell script
# make sure "make" and "docker" is usable
# "sudo" is used to run "docker"
make docker-build
```

Step 3: Run the docker image

```shell script
make docker-run
```

#### Method 2: Use Python in the Ubuntu (tested in 24.04 LTS)

Step 1: Create a Python 3.10 environment and activate it

```shell script
# example with conda
conda create --name captcha -y python=3.10
conda activate captcha
```

Step 2: Change pwd to the directory of this README.md

Step 3: Install the required packages

```shell script
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6 -y
pip install --upgrade pip
pip install -r ./requirements.txt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

Step 4: Execute main.py

```shell script
python main.py
```
