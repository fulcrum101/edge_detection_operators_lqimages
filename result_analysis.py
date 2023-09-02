import json
import numpy as np
def main():
    with open('data-output/result_scores.json') as file:
        data = json.load(file)
    final_results = {
        'Laplacian': {},
        'Canny': {},
        'Prewitt': {},
        'Sobel': {},
        'Scharr': {},
        'Robert': {},
        'PST': {}
    }
    for al in ['Laplacian', 'Canny', 'Prewitt', 'Sobel', 'Scharr', 'Robert', 'PST']:
        for case in ['blurred', 'scattered', 'noise', 'gamma_w', 'gamma_d', 'm_blurred_v', 'm_blurred_h']:
            for i in range(5):
                final_results[al][f'{case}_{i}'] = np.average(data[al][case][i])
            final_results[al][f'{case}_overall'] = np.average(data[al][case])
    with open('final_results.json', "w") as file:
        json.dump(final_results, file, indent=3)

if __name__ == '__main__':
    main()