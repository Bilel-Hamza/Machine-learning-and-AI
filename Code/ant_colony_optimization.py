from random import uniform, randint
from matplotlib import pyplot as plt
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage

import time
import utils


NUMBER_OF_ITERATIONS = 100  # number of iterations for the algorithm to run
EVAPORATION_FACTOR = 0.5 # number between 0 and 1; 0 for complete evaporation, 1 for no evaporation
UPDATED_PHEROMONE_VALUE = 4  # exact pheromone value for each ant travelled, initial value is 1
ALPHA = 1  # parameter to control the influence of previous trails on the edge
BETA = 1  # parameter to control the influence of the desirability of the edge


def iterate_ants(ants, edges, number_of_cities):
    trails1 = []
    for a in ants:
        # print('for ant', a)
        visited = [0] * number_of_cities
        trail = []
        count = 0
        q = a
        visited[q] = 1
        trail.append(q)
        while count < number_of_cities - 1:
            # print('on node ', q, ' ', end='')
            sum1 = 0
            numerator = []
            prob = []
            sub_visit = [0] * number_of_cities
            for v in range(len(visited)):
                if visited[v] == 1:
                    sub_visit[v] = 1
            for p in range(number_of_cities):
                if sub_visit[p] == 0:
                    this_edge = edges[(number_of_cities * p) + q]
                    ph = this_edge[2]
                    w = this_edge[1]
                    ph = ph ** ALPHA
                    w = w ** BETA
                    if w == 0:
                        num = float('inf')
                    else:
                        num = ph / w
                    numerator.append((p, num))
                    sum1 += num
            for v in range(len(numerator)):
                prob.append((numerator[v][0], numerator[v][1] / sum1))
            seed = uniform(0, 1)
            sum2 = 0
            temp = None
            for p in range(len(prob)):
                sum2 += prob[p][1]
                if sum2 < seed:
                    continue
                else:
                    temp = prob[p][0]
                    break
            visited[temp] = 1
            trail.append(temp)
            count += 1
            q = temp
        trails1.append(trail)
    return trails1


def update_edge(a, b, edges, number_of_cities):
    this1 = edges[(number_of_cities * a) + b]
    this2 = edges[(number_of_cities * b) + a]
    w1 = this1[1]
    w2 = this2[1]
    ph1 = this1[2]
    ph2 = this2[2]
    # print(w1, w2, ph1, ph2)
    edges[(number_of_cities * a) + b] = ((a, b), w1, ph1 + UPDATED_PHEROMONE_VALUE)
    edges[(number_of_cities * b) + a] = ((b, a), w2, ph2 + UPDATED_PHEROMONE_VALUE)


def update_pheromone_and_find_best_path(ts, nodes, edges, number_of_cities):
    min1 = float('inf')
    min_path1 = []
    for t in ts:
        sum1 = 0
        for i in range(len(t) - 1):
            a = t[i]
            b = t[i + 1]
            sum1 += utils.dist(a, b, nodes)
            update_edge(a, b, edges, number_of_cities)
        if min1 > sum1:
            min1 = sum1
            min_path1 = []
            min_path1 += t
    ret_path = min_path1[:]
    ret_path.append(min_path1[0])
    min1 += utils.dist(min_path1[-1], min_path1[0], nodes)
    return min1, ret_path


def evaporation(edges, number_of_cities):
    for e in edges:
        p1 = e[0][0]
        p2 = e[0][1]
        w = e[1]
        ph = e[2]
        edges[(number_of_cities * p1) + p2] = ((p1, p2), w, ph * EVAPORATION_FACTOR)


def execute(points, number_of_cities):
    start_time = time.time()
    counter2 = 0
    print()
    print("Ant Colony Optimization")
    nodes = points
    edges = []
    for x in range(number_of_cities):
        for y in range(number_of_cities):
            if x is not y:
                edges.append(((x, y), utils.dist(x, y, nodes), 1))
            else:
                edges.append(((x, y), float("inf"), 1))
    global_minima = float('inf')
    global_min_path = []
    for r in range(NUMBER_OF_ITERATIONS):
        ants = list(range(number_of_cities))
        trails = iterate_ants(ants, edges, number_of_cities)
        min_value, min_path = update_pheromone_and_find_best_path(trails, nodes, edges, number_of_cities)
        evaporation(edges, number_of_cities)
        if global_minima > min_value:
            global_minima = min_value
            global_min_path = min_path
        counter2 += 1
        print(counter2, ': ', min_value, '     ', min_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("Elapsed time:", elapsed_time, "seconds") 
    print()
    print('Best path from ant colony optimisation : ', global_min_path, 'shortest distance =', global_minima)
    print('Using these hyperparameters:  EVAPORATION FACTOR = ',  EVAPORATION_FACTOR,
        'Updated Pheromone value =',UPDATED_PHEROMONE_VALUE  )
    
    # Convert the best path (list) to a string
    best_path_str = ', '.join(map(str,global_min_path))   
    print(best_path_str) 
    data ={
        'Number Of Cities': [number_of_cities],
        'Best Path ACO' :  [best_path_str],
        'Global_minima' : [global_minima],
        'Evaporation Factor': [EVAPORATION_FACTOR],
        'Updated Pheromone Value': [UPDATED_PHEROMONE_VALUE],
        'CPU Time': [elapsed_time]
        }
    df =pd.DataFrame(data)
    print('')
    print(df)
    
    excel_file = 'results.xlsx'  # Specify the desired filename
    df.to_excel(excel_file, index=False)  # Set index=False to exclude the index column

    print(f"Results exported to {excel_file}")
    
    
    plot_x = []
    plot_y = []

    for p in global_min_path:
        plot_x.append(nodes[p][0])
        plot_y.append(nodes[p][1])

    plt.plot(plot_x, plot_y, 'ko-',color = 'cyan', linewidth=1.5, label='ant colony ' + "{:.2f}".format(global_minima))
    # Ajouter des points noirs uniquement
    plt.plot(plot_x, plot_y, 'ko') 
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='upper left')
    plt.title('ACO')
    plt.show()
    
    # Save the figure as an image
    image_file = 'plot.png'
    plt.savefig(image_file)# Save the plot as an image
    plt.clf()  # Clear the figure after saving


    # Load the Excel file using openpyxl
    workbook = openpyxl.load_workbook(excel_file)
    worksheet = workbook.active  # Get the active worksheet

    
    # Insert the image into the worksheet
    img = OpenpyxlImage(image_file)  # Create an Image object from the image file
    worksheet.add_image(img, 'J1')  # Adjust the cell reference as needed
    
    # Save the Excel file
    workbook.save(excel_file)
    print(f"Image inserted into {excel_file}")
    
    
    
    
