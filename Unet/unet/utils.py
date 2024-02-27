def add_mask(image, mask, class_colors):
    classes = mask.unique()
    classes = classes[classes>0].int()
    for cls_idx in classes:
        match = (mask == cls_idx).nonzero()

        image[:, match[:,0], match[:,1]] = image[:, match[:,0], match[:,1]] * 0.5 + class_colors[cls_idx] * 0.5
    return image