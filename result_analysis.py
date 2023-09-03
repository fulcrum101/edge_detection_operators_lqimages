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
            l = 0
            for i in range(5):
                final_results[al][f'{case}_{i}'] = np.average(data[al][case][i])
                l += np.average(data[al][case][i])*(i+1)
            final_results[al][f'{case}_overall'] = l/15
    with open('final_results.json', "w") as file:
        json.dump(final_results, file, indent=3)

if __name__ == '__main__':
    main()