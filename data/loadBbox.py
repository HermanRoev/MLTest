import random
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.patches as patches


def visualize_coco(image_id, coco):
    img_info = coco.loadImgs(image_id)[0]
    image = io.imread(img_info['coco_url'])

    plt.imshow(image); plt.axis('off')
    ann_ids = coco.getAnnIds(imgIds=img_info['id'])
    anns = coco.loadAnns(ann_ids)

    ax = plt.gca()

    for ann in anns:
        bbox = ann['bbox']
        rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        cat = coco.loadCats(ann['category_id'])[0]['name']
        plt.text(bbox[0], bbox[1] - 10, cat, color='red')

    plt.show()


# Load the data
coco = COCO('../../data/annotations/instances_train2017.json')

# Visualize a specific image
image_ids = list(coco.imgs.keys())  # Replace with the ID of the image you want to visualize
random_image_id = random.choice(image_ids)
visualize_coco(random_image_id, coco)
