# -*- coding: utf-8 -*-
# @Time    : 13/12/2018 12:44
# @Author  : weiziyang
# @FileName: run.py
# @Software: PyCharm

from K_neighbour import KNeighbour
from words_bag import WordsBag
from GIST import GIST

if __name__ == "__main__":
    with open('run_result/run1.txt', 'w') as f:
            run1 = KNeighbour(k=1, pic_size=8)
            run1.train()
            text = run1.predict_all()
            f.write(text)

    with open('run_result/run2.txt', 'w') as f:
        run2 = WordsBag(patch_gap=4, patch_size=8, cluster_num=400, output_patch_image=False,
                        mini_patch=True, pic_size=256, force_generate_again=False)
        run2.train()
        text = run2.predict_all()
        f.write(text)

    with open('run_result/run3.txt', 'w') as f:
        run3 = GIST(pic_size=(160, 160))
        run3.train()
        text = run3.predict_all()
        f.write(text)
