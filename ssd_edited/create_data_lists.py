from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(voc07_path='../TumorDetection/tumor_detect/input/brain-tumor-object-detection-datasets/axial_t1wce_2_class/',
                      voc12_path='../TumorDetection/tumor_detect/input/brain-tumor-object-detection-datasets/axial_t1wce_2_class/',
                      output_folder='./output')
