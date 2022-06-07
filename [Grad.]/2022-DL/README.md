# Simplified Neural Collaborative Filtering in PyTorch

This repository is simplified Neural Collaborative Filtering, which originally created by Xiangnan He et.al  <br>

Original work is : **Neural Collaborative Filtering(Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua)**

He, Xiangnan, et.al. “Neural Collaborative Filtering”. arXiv:1708.05031 [cs], 2017. arXiv.org, http://arxiv.org/abs/1708.05031.

And 2nd edit of original work is by [HarshdeepGupta](https://github.com/HarshdeepGupta/recommender_pytorch)

This is 3rd edition of original work, and 2nd edition of HarshdeepGupta's work.

## Prepare the dataset

You can download raw data file from Grouplens.

https://grouplens.org/datasets/movielens/latest/


And you can run below code for preparing train, test data.

`
python train_test_split.py
`


## Run the deep learning based Model

The following command runs the model and prints the metrics.

(Many other arguments are available, you can see inside MLP.py file.)

`
python MLP.py
`

Or, you can run below code (my recommendation :))

`
python MLP.py --epochs 20 --batch_size 256 --layers [100,50,20,10] --lr 1e-3
`


## Recommend to specific user

After training model (or with pre-trained model), you can see how recommendation works to specific user.

Use below notebook file to see how recommendation works.

`
MovieRecommender.ipynb
`