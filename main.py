import cv2, os, shutil, logging, json
import numpy as np

from scripts import load_image, detect_edges, blur_images, scatter_images, noise_images, gamma_d_images, gamma_w_images, motion_blur_h_images, motion_blur_v_images

SCORES = {
    'Laplacian' : {
        'blurred': [[], [], [], [], []],
        'scattered': [[], [], [], [], []],
        'm_blurred_h': [[], [], [], [], []],
        'm_blurred_v': [[], [], [], [], []],
        'noise': [[], [], [], [], []],
        'gamma_w': [[], [], [], [], []],
        'gamma_d': [[], [], [], [], []],
    },
    'Canny': {
        'blurred': [[], [], [], [], []],
        'scattered': [[], [], [], [], []],
        'm_blurred_h': [[], [], [], [], []],
        'm_blurred_v': [[], [], [], [], []],
        'noise': [[], [], [], [], []],
        'gamma_w': [[], [], [], [], []],
        'gamma_d': [[], [], [], [], []],
    },
    'Prewitt':{
        'blurred': [[], [], [], [], []],
        'scattered': [[], [], [], [], []],
        'm_blurred_h': [[], [], [], [], []],
        'm_blurred_v': [[], [], [], [], []],
        'noise': [[], [], [], [], []],
        'gamma_w': [[], [], [], [], []],
        'gamma_d': [[], [], [], [], []],
    },
    'Sobel': {
        'blurred': [[], [], [], [], []],
        'scattered': [[], [], [], [], []],
        'm_blurred_h': [[], [], [], [], []],
        'm_blurred_v': [[], [], [], [], []],
        'noise': [[], [], [], [], []],
        'gamma_w': [[], [], [], [], []],
        'gamma_d': [[], [], [], [], []],
    },
    'Scharr':{
        'blurred': [[], [], [], [], []],
        'scattered': [[], [], [], [], []],
        'm_blurred_h': [[], [], [], [], []],
        'm_blurred_v': [[], [], [], [], []],
        'noise': [[], [], [], [], []],
        'gamma_w': [[], [], [], [], []],
        'gamma_d': [[], [], [], [], []],
    },
    'Robert': {
        'blurred': [[], [], [], [], []],
        'scattered': [[], [], [], [], []],
        'm_blurred_h': [[], [], [], [], []],
        'm_blurred_v': [[], [], [], [], []],
        'noise': [[], [], [], [], []],
        'gamma_w': [[], [], [], [], []],
        'gamma_d': [[], [], [], [], []],
    },
    'PST': {
        'blurred': [[], [], [], [], []],
        'scattered': [[], [], [], [], []],
        'm_blurred_h': [[], [], [], [], []],
        'm_blurred_v': [[], [], [], [], []],
        'noise': [[], [], [], [], []],
        'gamma_w': [[], [], [], [], []],
        'gamma_d': [[], [], [], [], []],
    }
}

def main():
    if os.path.exists('data-output/logs.log'):
        os.remove('data-output/logs.log')
    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler('data-output/logs.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    for file_name in os.listdir('data-input'):
        # Create image folder
        img = load_image(file_name)
        img_folder_path = f'data-output/{file_name[:-4]}'
        if os.path.exists(img_folder_path):
            shutil.rmtree(img_folder_path)
        os.mkdir(img_folder_path)
        os.mkdir(f'{img_folder_path}/original/')
        # Lower quality of an image
        images = {
            'blurred': blur_images(img),
            'scattered': scatter_images(img),
            'noise': noise_images(img),
            'gamma_w': gamma_w_images(img),
            'gamma_d': gamma_d_images(img),
            'm_blurred_v': motion_blur_v_images(img),
            'm_blurred_h': motion_blur_h_images(img)
        }

        for key in images:
            os.mkdir(f'{img_folder_path}/{key}')
            for i, s in enumerate(images[key]):
                os.mkdir(f'{img_folder_path}/{key}/{i}')
                cv2.imwrite(f'{img_folder_path}/{key}/{i}/original.jpg', s)


        for al in ['Laplacian', 'Canny', 'Prewitt', 'Sobel', 'Scharr', 'Robert', 'PST']:
            original = detect_edges(al, img)
            cv2.imwrite(f'{img_folder_path}/original/{al}.jpg', original)
            for key in images:
                for i, l_img in enumerate(images[key]):
                    detected_edges = detect_edges(al, l_img)
                    cv2.imwrite(f'{img_folder_path}/{key}/{i}/{al}.jpg', detected_edges)
                    # Calculate metrics
                    intersect = cv2.bitwise_and(detected_edges, original)
                    union = cv2.bitwise_or(detected_edges, original)
                    if np.sum(intersect == 255) and np.sum(union == 255):
                        score = np.sum(intersect == 255) / np.sum(union == 255)
                    else:
                        score = 0.0
                    SCORES[al][key][i].append(score)
                    logger.info(f'{file_name} | {al} {key} {i} score: {score}')




    with open("data-output/result_scores.json", "w") as outfile:
        json.dump(SCORES, outfile, indent=3)

if __name__ == '__main__':
    main()