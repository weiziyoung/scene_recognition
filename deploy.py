# -*- coding: utf-8 -*-
# @Time    : 06/12/2018 22:36
# @Author  : weiziyang
# @FileName: deploy.py
# @Software: PyCharm
from K_neighbour import KNeighbour
from words_bag import WordsBag
from sift import SIFT
from GIST import GIST

import time
import logging

if __name__ == "__main__":
    # find the best parameter for run1
    # for k in range(1, 8):
    #     for size in (i*8 for i in range(1, 8)):
    #         k_neigh = KNeighbour(k=k, pic_size=size)
    #         k_neigh.train()
    #         mark = k_neigh.test_model()
    #         time.sleep(10)

    # find the best parameter for run2
    # for size in [32*2**i for i in range(0, 4)] + [None]:
    #     for gap in [4*_ for _ in range(1, 3)]:
    #         patch_size = gap * 2
    #         for cluster_num in [200*_ for _ in range(1, 6)]:
    #             try:
    #                 bag = WordsBag(patch_gap=gap, patch_size=patch_size, cluster_num=cluster_num,
    #                                output_patch_image=False, mini_patch=True, pic_size=size,
    #                                force_generate_again=True)
    #                 bag.train()
    #                 mark = bag.test_model()
    #             except Exception as e:
    #                 continue
    #
    # # find the best parameter for run3:
    for cluster in [100*i for i in range(1, 11)]:
        try:
            sift = SIFT(cluster_num=cluster, mini_patch=True, force_generate_again=True)
            sift.train()
            mark = sift.test_model()
        except Exception as e:
            continue

    # find the best parameter for gist
    # for img_size in [32*i for i in range(1, 9)]:
    #     gist = GIST(pic_size=(img_size, img_size))
    #     gist.train()
    #     gist.test_model()

