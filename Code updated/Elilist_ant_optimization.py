from random import uniform
from matplotlib import pyplot as plt
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
import time
import utils

NUMBER_OF_ITERATIONS = 100  # number of iterations for the algorithm to run
EVAPORATION_FACTOR = 0.6  # evaporation coefficient between 0 and 1; closer to 0 means faster evaporation
UPDATED_PHEROMONE_VALUE = 7  # pheromone value to add for each ant that travels a path
ALPHA = 1  # influence of pheromone
BETA = 1  # influence of distance desirability
ELITE_FACTOR = 2  # factor to amplify pheromone updating for elite ants


def iterate_ants(ants, edges, number_of_cities):
    trails1 = []
    for a in ants:
        visited = [0] * number_of_cities
        trail = []
        count = 0
        q = a
        visited[q] = 1
        trail.append(q)
        while count < number_of_cities - 1:
            sum1 = 0
            numerator = []
            prob = []
            for p in range(number_of_cities):
                if visited[p] == 0:
                    this_edge = edges[(number_of_cities * p) + q]
                    ph = this_edge[2]
                    w = this_edge[1]
                    ph = ph ** ALPHA
                    w = w ** BETA
                    num = ph / w if w != 0 else float('inf')
                    numerator.append((p, num))
                    sum1 += num
                    
            for v in range(len(numerator)):
                prob.append((numerator[v][0], numerator[v][1] / sum1))
            seed = uniform(0, 1)
            sum2 = 0
            
            for p in range(len(prob)):
                sum2 += prob[p][1]
                if sum2 >= seed:
                    temp = prob[p][0]
                    break

            visited[temp] = 1
            trail.append(temp)
            count += 1
            q = temp
        trails1.append(trail)
    return trails1


def update_edge(a, b, edges, number_of_cities, elite=False):
    this1 = edges[(number_of_cities * a) + b]
    this2 = edges[(number_of_cities * b) + a]
    ph_increase = UPDATED_PHEROMONE_VALUE * (ELITE_FACTOR if elite else 1)
    edges[(number_of_cities * a) + b] = ((a, b), this1[1], this1[2] + ph_increase)
    edges[(number_of_cities * b) + a] = ((b, a), this2[1], this2[2] + ph_increase)


def update_pheromone_and_find_best_path(ts, nodes, edges, number_of_cities):
    min1 = float('inf')
    min_path1 = []
    
    for t in ts:
        sum1 = 0
        for i in range(len(t) - 1):
            a = t[i]
            b = t[i + 1]
            sum1 += utils.dist(a, b, nodes)
            update_edge(a, b, edges, number_of_cities, elite=False)  # Normal update for all ants
            
        if min1 > sum1:
            min1 = sum1
            min_path1 = t
            elite_path = t  # Store the best path to update directions heavily
            
    # Update pheromones for the best path found this iteration
    for i in range(len(elite_path) - 1):
        a = elite_path[i]
        b = elite_path[i + 1]
        update_edge(a, b, edges, number_of_cities, elite=True)  # Heavier update for elite path

    ret_path = min_path1[:]
    ret_path.append(min_path1[0])
    min1 += utils.dist(min_path1[-1], min_path1[0], nodes)
    return min1, ret_path


def evaporation(edges, number_of_cities):
    for e in edges:
        p1, p2 = e[0]
        ph = e[2]
        edges[(number_of_cities * p1) + p2] = ((p1, p2), e[1], ph * EVAPORATION_FACTOR)


def execute(points, number_of_cities):
    start_time = time.time()
    print()
    print("Ant Colony Optimization with Elite Strategy")
    nodes = points
    edges = []
    
    for x in range(number_of_cities):
        for y in range(number_of_cities):
            if x != y:
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
        
        print(f"Iteration {r + 1}: Shortest path = {min_value}, Path = {min_path}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("Elapsed time:", elapsed_time, "seconds") 
    print()
    print('Best path from ant colony optimization: ', global_min_path, 'shortest distance =', global_minima)
    print('Using these hyperparameters:  EVAPORATION FACTOR = ', EVAPORATION_FACTOR,
          'Updated Pheromone value =', UPDATED_PHEROMONE_VALUE, 'Elite Factor =', ELITE_FACTOR)
    
    best_path_str = ', '.join(map(str, global_min_path))
    print(best_path_str) 
    data = {
        'Number Of Cities': [number_of_cities],
        'Best Path ACO': [best_path_str],
        'Global_minima': [global_minima],
        'Evaporation Factor': [EVAPORATION_FACTOR],
        'Updated Pheromone Value': [UPDATED_PHEROMONE_VALUE],
        'Elite Factor': [ELITE_FACTOR],
        'CPU Time': [elapsed_time]
    }
    df = pd.DataFrame(data)
    print('')
    print(df)
    
    excel_file = 'results.xlsx'
    df.to_excel(excel_file, index=False)

    print(f"Results exported to {excel_file}")
    
    city_names = [f"City {i}" for i in range(number_of_cities)]  # Example city names

    plot_x = []
    plot_y = []

    for p in global_min_path:
        plot_x.append(nodes[p][0])
        plot_y.append(nodes[p][1])

    plt.plot(plot_x, plot_y, 'ko-', color='brown', linewidth=1.5, label='Ant-Elite Strategy ' + "{:.2f}".format(global_minima))
    
    plt.plot(plot_x, plot_y, 'ko')

    for i, city in enumerate(global_min_path):
        plt.text(nodes[city][0], nodes[city][1], city_names[city], fontsize=9, ha='right')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='upper left')
    plt.title('ACO with Elite Strategy')
    plt.show()
    
    image_file = 'plot.png'
    plt.savefig(image_file)
    plt.clf()

    workbook = openpyxl.load_workbook(excel_file)
    worksheet = workbook.active

    img = OpenpyxlImage(image_file)
    worksheet.add_image(img, 'J1')
    
    workbook.save(excel_file)
    print(f"Image inserted into {excel_file}")


