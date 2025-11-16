# AEDM

This is the code for the algorithm proposed by our paper:

"Chi Hong, Jiyue Huang, Lydia Y. Chen, and Robert Birke. "Adversarial Knowledge
Extraction via Steering Diffusion Models." In the 31st International Conference on Neural
Information Processing, 2024."

This project relies on https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main to implement diffusion models. To facilitating users, we provide a copy in this repo.

# Before running
To run the algorithm, please extract the pretrained diffusion model "trained_models/diffusion_models". Please use the command
- sudo apt install p7zip-full
- cd trained_models/diffusion_models/
- 7z x my_zip.7z.001

Then you will get the pretrained diffusion model on imagenet, and you can run the experiments. You may replace the pretrained models by yours.

An example of running the algorithm is shown in "run.py".

# To run this file
The project is developed under python 3.8.10

- pip install -r requirements.txt
- python run.py

# Expected Results
After running the example "run.py", we can get the following expected Results. Please note that due to randomness, the final results you have may differ slightly from what is shown here.
- You will see the target model accuracy is 96.59 %.
- The accuracy of the substitute model will increaces with training over multiple epochs
- one training log is shown in "./cifar10_queries_100000.log"