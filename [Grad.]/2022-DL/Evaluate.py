"""
Evaluate the performance of top-k recommendation.
 > protocol : leave-one-out evaluation
 > measure : Hit-Ratio & NDCG

More details : Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback, SIGIR'16

@autor : HeXiangnan
@1st modifier : Harshdeep Gupta
@2nd modifier : Charlie Kim
"""
import math
import heapq # For getting top-K // Similar to java's priority queue
import numpy as np
from Dataset import MovieLensDataset

# Global variables for sharing across processes.
_model = None
_testRatings = None
_testNegatives = None
_topk = None

# Calculate Hit-Ratio
def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

# Calculate NDCG
def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

# Evaluate for one rating
def eval_one_rating(idx, full_dataset:MovieLensDataset):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]

    gtItem = rating[1]

    items.append(gtItem)

    # Calculate prediction score
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32') # u 값으로 item 수 만큼의 사용자 배열 생성

    feed_dict = {'user_id':users, 'item_id':np.array(items)}
    predictions = _model.predict(feed_dict)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_topk, map_item_score, key=map_item_score.get) # heapq를 이용, score가 가장 높은 원소 get
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)

    return (hr, ndcg)

# Evaluate performance (hit_ratio, ndcg) of model's top-k recommendation on test dataset
def evaluate_model(model, full_dataset:MovieLensDataset, topK:int):
    global _model
    global _testRatings
    global _testNegatives
    global _topk

    _model = model
    _testRatings = full_dataset.testRatings
    _testNegatives = full_dataset.testNegatives
    _topk = topK

    hits, ndcgs = [], []
    for i in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(i, full_dataset)
        hits.append(hr)
        ndcgs.append(ndcg)
    
    return (hits, ndcgs)
