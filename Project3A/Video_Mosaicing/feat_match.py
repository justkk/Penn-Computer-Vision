'''
  File name: feat_match.py
  Author:
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''


from annoy import AnnoyIndex

import numpy as np


THRES = 0.6

def feat_match(descs1, descs2):
  # Your Code Here

  descs1T = np.transpose(descs1)
  descs2T = np.transpose(descs2)

  an = AnnoyIndex(descs1.shape[0])

  for vecIndex in range(descs2T.shape[0]):
    an.add_item(vecIndex, descs2T[vecIndex, :])

  an.build(200)

  mathingIndex = []

  for vecIndex in range(descs1T.shape[0]):
    vector = descs1T[vecIndex, :];
    neighbours, distance = an.get_nns_by_vector(vector, 2, search_k=-1, include_distances=True)

    if distance[1]!= 0 and(1.0*distance[0])/distance[1] <= THRES:
      mathingIndex.append(neighbours[0])
    else:
      mathingIndex.append(-1)


  return np.array(mathingIndex)