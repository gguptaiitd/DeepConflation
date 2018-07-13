class mdict(dict):
    def __setitem__(self, key, value):
        """add the given value to the list of values for this key"""
        self.setdefault(key, []).append(value)

image_size = 32 * 1
image_w = 32
image_h = 1

def image(image_array):
    return cv2.resize(np.asarray(image_array), (image_w, image_h)).astype('float32') / 255


def generateHotVectorEncoding(data):
    i_data = array(data[2]).reshape((image_size))
    label = LabelEncoder().fit_transform(i_data)
    encoded = to_categorical(label)
    return [data[0], data[1], encoded.reshape((image_w,image_h,4))]

# Creates the dataset
def getDataset(display=True):
    image_fname = "/home/genome/som_data/refseqImages_" + str(image_size) + ".txt"
    data = []
    hot_data = []
    with open(image_fname) as image_file:
        for line in image_file:
            d = eval(line)
            data.append(d)
            hot_data.append(generateHotVectorEncoding(d))
    ref_image_data = {}
    mut_image_data = mdict()
    # image_label = []
    for d in hot_data:
    #for d in data:
        if d[1] != 0:
            key = d[0]
            mut_image_data[key] = d[2]
        else:
            key = d[0]
            ref_image_data[key] = d[2]