import numpy as np


class PatchMatch(object):
    def __init__(self, src, dst, patch_size, step=1, init_nnf=None, src_g=None, dst_g=None, g_alpha=0.5,
                 cal_gradient=False):
        self.src = src
        self.dst = dst
        self.patch_size = patch_size
        self.nnf = init_nnf
        self.dist_map = {}
        self.step = step

        self.src_g = src_g
        self.dst_g = dst_g
        self.g_alpha = g_alpha
        self.cal_gradient = cal_gradient

        self.initialization()

    def nnf_valid(self, pos):
        return pos[0] >= 0 and pos[0] < self.dst.shape[0] and pos[1] >= 0 and pos[1] < self.dst.shape[1]

    def cal_distance(self, a, b):
        if (a[0], a[1], b[0], b[1]) in self.dist_map:
            return self.dist_map[(a[0], a[1], b[0], b[1])]

        p = self.patch_size // 2

        dx0 = min(a[0], b[0], p)
        dx1 = min(self.src.shape[0] - a[0], self.dst.shape[0] - b[0], p + 1)
        dy0 = min(a[1], b[1], p)
        dy1 = min(self.src.shape[1] - a[1], self.dst.shape[1] - b[1], p + 1)
        num = (dx0 + dx1) * (dy0 + dy1)

        patch_a = self.src[a[0] - dx0:a[0] + dx1, a[1] - dy0:a[1] + dy1]
        patch_b = self.dst[b[0] - dx0:b[0] + dx1, b[1] - dy0:b[1] + dy1]
        temp = patch_b - patch_a
        dist = np.sum(np.square(temp)) / num

        if self.cal_gradient:
            patch_a_g = self.src_g[a[0] - dx0:a[0] + dx1, a[1] - dy0:a[1] + dy1]
            patch_b_g = self.dst_g[b[0] - dx0:b[0] + dx1, b[1] - dy0:b[1] + dy1]
            temp_g = patch_b_g - patch_a_g
            dist_g = np.sum(np.square(temp_g)) / num
            dist = dist * (1 - self.g_alpha) + dist_g * self.g_alpha

        self.dist_map[(a[0], a[1], b[0], b[1])] = dist
        return dist

    def initialization(self):
        src_h = np.size(self.src, 0)
        src_w = np.size(self.src, 1)
        dst_h = np.size(self.dst, 0)
        dst_w = np.size(self.dst, 1)

        if self.nnf is None:
            self.nnf = np.zeros([src_h, src_w, 2], dtype=np.int)
            self.nnf[:, :, 0] = np.random.randint(0, dst_h - 1, [src_h, src_w])
            self.nnf[:, :, 1] = np.random.randint(0, dst_w - 1, [src_h, src_w])

        for i in range(src_h):
            for j in range(src_w):
                self.cal_distance([i, j], self.nnf[i, j])

    def reconstruction(self):
        src_h = np.size(self.src, 0)
        src_w = np.size(self.src, 1)
        temp = np.zeros_like(self.src)
        for i in range(src_h):
            for j in range(src_w):
                temp[i, j] = self.dst[self.nnf[i, j][0], self.nnf[i, j][1]]
        return temp

    def propagation(self, a, is_odd):
        src_h = np.size(self.src, 0)
        src_w = np.size(self.src, 1)
        x = a[0]
        y = a[1]
        pos_list = [a]
        dist_list = [self.cal_distance(a, self.nnf[a[0], a[1]])]
        if is_odd:
            left = np.array([max(x - self.step, 0), y])
            up = np.array([x, max(y - self.step, 0)])
            left_p = a + self.nnf[left[0], left[1]] - left
            up_p = a + self.nnf[up[0], up[1]] - up
            if self.nnf_valid(left_p):
                pos_list.append(left_p)
                dist_list.append(self.cal_distance(a, left_p))
            if self.nnf_valid(up_p):
                pos_list.append(up_p)
                dist_list.append(self.cal_distance(a, up_p))
        else:
            right = np.array([min(x + self.step, src_h - 1), y])
            down = np.array([x, min(y + self.step, src_w - 1)])
            right_p = a + self.nnf[right[0], right[1]] - right
            down_p = a + self.nnf[down[0], down[1]] - down
            if self.nnf_valid(right_p):
                pos_list.append(right_p)
                dist_list.append(self.cal_distance(a, right_p))
            if self.nnf_valid(down_p):
                pos_list.append(down_p)
                dist_list.append(self.cal_distance(a, down_p))

        idx = np.argmin(dist_list)
        if idx > 0:
            self.nnf[x, y] = pos_list[idx]

    def random_search(self, a, alpha=0.5):
        x = a[0]
        y = a[1]
        dst_h = np.size(self.dst, 0)
        dst_w = np.size(self.dst, 1)
        i = 4
        search_h = dst_h * alpha ** i
        search_w = dst_w * alpha ** i
        b_x = self.nnf[x, y][0]
        b_y = self.nnf[x, y][1]
        while search_h > 1 and search_w > 1:
            search_min_r = max(b_x - search_h, 0)
            search_max_r = min(b_x + search_h, dst_h)
            random_b_x = np.random.randint(search_min_r, search_max_r)
            search_min_c = max(b_y - search_w, 0)
            search_max_c = min(b_y + search_w, dst_w)
            random_b_y = np.random.randint(search_min_c, search_max_c)
            b = [random_b_x, random_b_y]
            if self.cal_distance(a, b) < self.cal_distance(a, self.nnf[a[0], a[1]]):
                self.nnf[x, y] = b
            i += 1
            search_h = dst_h * alpha ** i
            search_w = dst_w * alpha ** i

    def NNS(self, itr=5):
        src_h = np.size(self.src, 0)
        src_w = np.size(self.src, 1)

        for itr in range(1, itr + 1):
            print("iteration: %d" % (itr))
            if itr % 2 == 0:
                for i in range(src_h - 1, -1, -self.step):
                    for j in range(src_w - 1, -1, -self.step):
                        a = [i, j]
                        self.propagation(a, False)
                        self.random_search(a)
            else:
                for i in range(0, src_h, self.step):
                    for j in range(0, src_w, self.step):
                        a = [i, j]
                        self.propagation(a, True)
                        self.random_search(a)
        return self.nnf, self.dist_map


if __name__ == '__main__':
    import sys
    import cv2 as cv

    nameA = sys.argv[1]
    nameB = sys.argv[2]
    nameC = sys.argv[3]

    imageA = cv.imread(nameA)
    imageB = cv.imread(nameB)

    patchmatch = PatchMatch(imageA, imageB, 5)
    patchmatch.NNS();
    imageC = patchmatch.reconstruction()
    cv.imwrite(nameC,imageC)

