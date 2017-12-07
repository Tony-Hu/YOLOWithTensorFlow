import matplotlib.pyplot as plt
import matplotlib.patches as patches
from VOC2012ImageReader import ImageReader
x = ImageReader.RESIZED_PIC_SIZE[0]
y = ImageReader.RESIZED_PIC_SIZE[1]


class ImageShower:
    counter = 1

    @staticmethod
    def show_image_and_save(image, boxes, save=True):
        fig, ax = plt.subplots()
        print()
        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                box = boxes[i][j]
                if box[0] == 0:
                    continue
                ax.add_patch(
                    patches.Rectangle(
                        ((box[1] - box[3] / 2) * x, (box[2] - box[4] / 2) * y),    # (x,y)
                        box[3] * x,    # width
                        box[4] * y,    # height
                        fill=False, color='red'))
                plt.text(box[1] * x, (box[2] + box[4] / 2) * y + 10, ImageShower.__get_label(box), color='red')
        ax.imshow(image)
        plt.show()
        if save:
            fig.savefig('output' + str(ImageShower.counter) + ".png", format='png')
        ImageShower.counter += 1
        print(((box[1] - box[3] / 2) * x, (box[2] - box[4] / 2) * y, box[3] * x, box[4] * y))

    @staticmethod
    def __get_label(box):
        for i in range(6, len(box)):
            if box[i] == 1:
                return ImageReader.CLASSES[i - 5]
        return 'unknown'