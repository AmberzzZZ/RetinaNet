import numpy as np
import os


class Retina_Kmeans:

    def __init__(self, label_dir):
        self.label_dir = label_dir
        self.input_size = (512,512)   # wh
        self.anchor_sizes = [32,64,128,256,512]
        self.iou_threshold = 0.5
        self.num_anchor_ratios = 9   # n_anchors for each level

    def txt2boxes(self):
        boxes = []
        for file in os.listdir(self.label_dir):
            if 'txt' not in file:
                continue
            f = open(os.path.join(self.label_dir, file), 'r')
            for line in f.readlines():
                cls, xc, yc, w, h = map(float, line.strip().split(' '))   # normed
                boxes.append([w, h])
            f.close()
        boxes = np.array(boxes)
        return boxes

    def txt2yolo_clusters(self):
        boxes = self.txt2boxes()
        # boxes = np.random.uniform(0.2, 0.8, (200,2))   # for test
        clusters = self.kmeans(boxes, n_clusters=self.num_anchor_ratios)
        clusters = clusters[np.lexsort(clusters.T[0, None])]
        print("K anchors:\n {}".format(clusters))
        print("Accuracy: {:.2f}%".format(self.avg_iou(boxes, clusters) * 100))
        return clusters

    def txt2retina_ratios(self):
        clusters = self.txt2yolo_clusters()
        ratios = clusters / np.sqrt(clusters.prod(axis=1, keepdims=True))
        ratios = ratios.round(1).reshape((1,-1,2))     # [1,N,2]
        anchors_sizes = np.array(self.anchor_sizes).reshape(-1, 1, 1)
        anchors = (ratios * anchors_sizes).reshape(-1,2)
        print("ratios: ", ratios.reshape(-1,2))
        print("anchors: ", anchors.shape)
        self.result2txt(ratios.reshape(-1,2))
        return anchors

    def kmeans(self, boxes, n_clusters=3, dist=np.median):
        # clusters: (N,2)
        box_number = boxes.shape[0]
        distances = np.empty((box_number, n_clusters))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(box_number, n_clusters, replace=False)]    # init k clusters

        while True:
            distances = 1 - self.iou(boxes, clusters)
            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(n_clusters):
                clusters[cluster] = dist(boxes[current_nearest==cluster], axis=0)
            last_nearest = current_nearest

        return clusters

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.num_anchor_ratios

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def result2txt(self, data):
        f = open("retina_ratios.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%.1f,%.1f" % (data[i][0], data[i][1])
            else:
                x_y = ", %.1f,%.1f" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()


if __name__ == "__main__":

    label_dir = "data/"
    kmeans = Retina_Kmeans(label_dir)
    kmeans.txt2retina_ratios()



