import cv2
import numpy
trainer = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)
test = cv2.imread("test_digits.png", cv2.IMREAD_GRAYSCALE)
rows = numpy.vsplit(trainer, 50)
cells = []
for row in rows:
    row_cells = numpy.hsplit(row, 50)
    for cell in row_cells:
        cell = cell.flatten()
        cells.append(cell)
cells = numpy.array(cells, dtype=numpy.float32)
# cells.base
k = numpy.arange(10)
cells_labels = numpy.repeat(k, 250)
test_digits = numpy.vsplit(test, 50)
# print(test_digits)
test_cells = []
for d in test_digits:
    d = d.flatten()
    test_cells.append(d)
# print(test_cells)
test_cells = numpy.array(test_cells, dtype=numpy.float32)
knn = cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
ret, result, neighbours, dist = knn.findNearest(test_cells, k=3)
print(result)
