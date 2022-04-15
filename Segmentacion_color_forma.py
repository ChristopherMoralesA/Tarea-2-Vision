#Importar las librerías por utilizar
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
from skimage import io
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.color import rgb2hsv,rgb2gray
from skimage.draw import circle_perimeter
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from matplotlib import cm



#PARAMETROS
#numero de colores incluyendo el fondo
N_COLORS = 5

#suma minima del fondo (blanco)
SUM_WHITE = 762 

# 8-vecindad
VECINDAD = [
    (-1,-1), (-1, 0), (-1, 1),
    ( 0,-1),          ( 0, 1),
    ( 1,-1), ( 1, 0), ( 1, 1)]

#Maxima desviacion de color (condicion de similitud)
MAX_SUM_COLOR = 80

#Numero de semillas para segmentar por color
N_SEEDS = 760

#Angulos característicos de cada figura
THETA_TRIANGULO = np.array([-np.pi/6,np.pi/6,np.pi/2],dtype=np.double)
THETA_CUADRADO = np.array([0,np.pi/2,-np.pi/2],dtype=np.double)
THETA_FIGS = np.array([THETA_TRIANGULO,THETA_CUADRADO],dtype=np.double)

#Angulos característicos de cada figura
THETA_TRIANGULO2 = np.array([-np.pi/6,np.pi/6],dtype=np.double)
THETA_CUADRADO2 = np.array([0,0],dtype=np.double)
THETA_FIGS2 = np.array([THETA_TRIANGULO2,THETA_CUADRADO2],dtype=np.double)

#Angulos que cada figura NO deberia tener
GHOST_THETA_TRIANGULO = np.array([0,np.pi/4],dtype=np.double)
GHOST_THETA_CUADRADO = np.array([-np.pi/6,np.pi/6],dtype=np.double)
GHOST_THETA_FIGS = np.array([GHOST_THETA_TRIANGULO,GHOST_THETA_CUADRADO],dtype=np.double)

#Quantificacion 4 colores
def img_to_5_colors(img):
    # Copia de la imagen por cuantificar
    img_5_colors = img

    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    img_5_colors = np.array(img_5_colors, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(img_5_colors.shape)
    assert d == 3
    image_array = np.reshape(img_5_colors, (w * h, d))
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=N_COLORS, random_state=0).fit(image_array_sample)
    colores = np.rint(np.array(kmeans.cluster_centers_, dtype=np.float64) *255)
    colores = colores.tolist()
    for color in colores:
        if color[0]+color[1]+color[2] >= SUM_WHITE:
            colores.remove(color)
    colores = np.array(colores,dtype=np.int32)
    return colores

# Condicion de similitud
def simililares(pix,color):
    sum = abs(pix[0]-color[0])+abs(pix[1]-color[1])+abs(pix[2]-color[2])
    return sum < MAX_SUM_COLOR

# Crecimiento de region
def region_growth(seed,img,out_img,visitado):  
    h, w = shape[:2]
    region = [seed]
    # Identifica el color de la semilla si no esta visitada
    x = seed[0]
    y = seed[1]
    seed_color = np.zeros((3),dtype=np.uint16)
    if (not visitado[y][x]):
        for COLOR in COLORES:
            if simililares(img[y][x],COLOR):
                seed_color = COLOR
                out_img[y][x] = seed_color
    else:
        seed_color = np.zeros((3),dtype=np.uint16)
        return out_img,seed_color
    # Crecimiento de la región
    while len(region):
        seed = region.pop(0)
        x = seed[0]
        y = seed[1]
        # Marca la semilla como visitada
        visitado[y][x] = 1
        for vecino in VECINDAD:
            cur_x = x + vecino[0]
            cur_y = y + vecino[1]
            # limites de la imagen
            if cur_x <0 or cur_y<0 or cur_x >= w or cur_y >=h :
                continue
            # crea nueva semilla si el pixel actual es similar a la semilla
            # y si aun no está visitado, marca pixel actual como visitado
            if (not visitado[cur_y][cur_x]) and simililares(img[cur_y][cur_x],seed_color):
                out_img[cur_y][cur_x] = seed_color
                visitado[cur_y][cur_x] = 1
                region.append((cur_x,cur_y))
    return out_img,seed_color

#Generacion de semillas
def generate_seeds():
    seeds = np.zeros((N_SEEDS,2), dtype=np.uint16)
    for i in range(N_SEEDS):
        seeds[i][0] = np.random.randint(0,shape[1])
        seeds[i][1] = np.random.randint(0,shape[0])
    return seeds

#Eliminar imagenes vacias
def clean_list_by_color(regs,regs_clr):
    indexes = []
    for i in range(N_SEEDS):
        if regs_clr[i].all() != 0:
            indexes.append(i)
    new_regs = np.zeros((len(indexes),shape[0],shape[1],shape[2]), dtype=np.int32)
    new_regs_clr = np.zeros((len(indexes),3), dtype=np.uint8)
    for i in range(len(indexes)):
        new_regs[i] = regs[indexes[i]]
        new_regs_clr[i] = regs_clr[indexes[i]]
    return new_regs,new_regs_clr

#Segmentacion por color
def segmentacion_por_color(img):
    SEEDS=generate_seeds()
    regions = np.zeros((N_SEEDS,shape[0],shape[1],shape[2]), dtype=np.int32)
    regions_colors = np.zeros((N_SEEDS,3), dtype=np.uint8)
    visitado = np.zeros(shape=(shape[0],shape[1],1), dtype=np.uint8)
    for i in range(N_SEEDS):
        out_img_temp = np.zeros(shape=shape, dtype=np.uint8)
        regions[i],regions_colors[i] = region_growth(SEEDS[i],img,out_img_temp,visitado)
    return clean_list_by_color(regions, regions_colors)

#Comparar dos colores
def same_color(color1,color2):
    if (color1[0]==color2[0]) and (color1[1]==color2[1]) and (color1[2]==color2[2]):
        return True
    else:
        return False

# Segmentacion por forma 1
def segmentacion_por_forma(regions):
    regs_form = []
    for i in range(len(regions)):
        image = rgb2gray(regions[i])*255**3/2
        edges = canny(image,0.5)
        done = False
        for j in range(len(THETA_FIGS)):
            lines = probabilistic_hough_line(edges, threshold=15, line_length=10,
                                        line_gap=3,theta=THETA_FIGS[j])
            ghost_lines = probabilistic_hough_line(edges, threshold=15, line_length=10,
                                        line_gap=3,theta=GHOST_THETA_FIGS[j])
            if (len(lines) >= 1) and (len(ghost_lines) == 0):
                if j==0:
                    regs_form.append('triangulo')
                    done = True
                if j==1:
                    regs_form.append('cuadrado')
                    done = True
        if done != True:
            regs_form.append('circulo')
    np.array(regs_form)
    return regs_form

# Segmentacion por forma 2
def segmentacion_por_forma2(regions):
    regs_form = []
    for i in range(len(regions)):
        image = rgb2gray(regions[i])*255**3/2
        edges = canny(image,0.5)
        done = False
        for j in range(len(THETA_FIGS2)):
            lines = probabilistic_hough_line(edges, threshold=10, line_length=30,
                                        line_gap=3,theta=THETA_FIGS2[j])
            if (len(lines) >= 1):
                if j==0:
                    regs_form.append('triangulo')
                    done = True
                if j==1:
                    regs_form.append('cuadrado')
                    done = True
        if done != True:
            regs_form.append('circulo')
    np.array(regs_form)
    return regs_form

# Segun el color y forma deseado crea una lista con las regiones que cumplen as caracteristicas
def identificador(regions,regions_clr,regions_form,wcolor='any',wform='any'):
    final_list = []
    final_list_clr = []
    final_list_form = []
    for i in range(len(regions)):
        if (same_color(regions_clr[i],wcolor) or wcolor=='any') and (regions_form[i]==wform or wform=='any'):
            final_list.append(regions[i])
            final_list_clr.append(regions_clr[i])
            final_list_form.append(regions_form[i])
    return np.array(final_list),np.array(final_list_clr),np.array(final_list_form)

#Obtener la direccion de las imagenes
path = os.getcwd()

#Imagen original
file_path = os.path.join(path , 'Imagen_a_utilizar.jpeg')
img_org = io.imread(file_path)
shape = img_org.shape
COLORES=img_to_5_colors(img_org)
DICT_COLOR = {'azul':COLORES[0],'amarillo':COLORES[1],'verde':COLORES[2],'rojo':COLORES[3]}
SEEDS=generate_seeds()
regs,regs_clr = segmentacion_por_color(img_org)
formas = segmentacion_por_forma(regs)
formas2 = segmentacion_por_forma2(regs)

segm1,segm1_clr,segm1_form = identificador(regs,regs_clr,formas,'any','cuadrado')
img_sum = np.zeros(shape=(shape[0],shape[1],3), dtype=np.uint8)
for i in range(segm1_clr.shape[0]):
    img_sum = img_sum + segm1[i]
plt.figure(figsize = (18,24))
plt.imshow(img_sum)
plt.axis('off')
plt.savefig('Segmentacion 1.jpg', bbox_inches='tight')

img_sum = np.zeros(shape=(shape[0],shape[1],3), dtype=np.uint8)
segm2,segm2_clr,segm2_form = identificador(regs,regs_clr,formas,DICT_COLOR['rojo'],'triangulo')
for i in range(segm2_clr.shape[0]):
    img_sum = img_sum + segm2[i]
plt.figure(figsize = (18,24))
plt.imshow(img_sum)
plt.axis('off')
plt.savefig('Segmentacion 2.jpg', bbox_inches='tight')

img_sum = np.zeros(shape=(shape[0],shape[1],3), dtype=np.uint8)
segm3,segm3_clr,segm3_form = identificador(regs,regs_clr,formas,DICT_COLOR['verde'],'any')
for i in range(segm3_clr.shape[0]):
    img_sum = img_sum + segm3[i]
plt.figure(figsize = (18,24))
plt.imshow(img_sum)
plt.axis('off')
plt.savefig('Segmentacion 3.jpg', bbox_inches='tight')
