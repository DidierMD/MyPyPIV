#PIV por Didier Muñoz

# La forma de usar este programa es:

# python3 PIV.py imagen1 imagen2 zonas_ancho zonas_alto archivo_salida

# Por ejemplo:

# python3 PIV.py imgs/006_b.png imgs/006_c.png 50 50 disp.png

from PIL import Image
import numpy as np
from matplotlib import pyplot
from numba import jit
import sys
import multiprocessing as mp
import time

def load_image( infilename ) :
    img = Image.open( infilename )
    img2 = img.convert("L")
    data = np.asarray( img2, dtype="float64" )
    return data

def split_image(img, zones_w, zones_h):
    v_split_points = np.arange(start=0, stop=len(img), step=zones_h)
    v_pieces = np.vsplit(img, v_split_points[1:])
    h_split_points = np.arange(start=0, stop=len(img[0]), step=zones_w)
    pieces = list()
    for r in v_pieces:
        pieces_row = np.hsplit(r, h_split_points[1:])
        pieces = pieces + pieces_row
    return pieces, len(v_pieces), len(pieces_row)

@jit
def fft_correlation(zone0, zone1):
    zone0_fft = np.fft.rfft2(zone0, norm='ortho')
    zone1_fft = np.fft.rfft2(zone1, norm='ortho')
    fft_mult = np.conjugate(zone0_fft) * zone1_fft
    corr = np.fft.irfft2(fft_mult, s=zone0.shape, norm='ortho')
    corr[0][0] = 0.
    return corr

@jit
def compute_correlation_maps(zones1, zones2):
    correlation_maps = list()
    for i in range(len(zones1)):
        corr = fft_correlation(zones1[i], zones2[i])
        correlation_maps.append(corr)
    return correlation_maps

@jit
def get_displacements(corr_maps):
    idx_lst = list()
    for k in range(len(corr_maps)):
        flat_idx = np.argmax(corr_maps[k])
        z_w = corr_maps[k].shape[0]
        z_h = corr_maps[k].shape[1]
        calto = z_h % 2
        cancho = z_w % 2
        i = flat_idx // z_w
        j = flat_idx % z_w
        if(i<=z_h//2):
            y = i-1+calto
        else:
            y = i-1-z_h+calto
        if(j <= z_w//2):
            x = j-1+cancho
        else:
            x = j-1-z_w+cancho
        idx_lst.append(np.array([x, y]))
    return idx_lst

# We need to transform each displacement to make the arrows start in the center of the region
@jit
def get_arrows_coordinates(idx_lst, corr_maps):
    coords_lst = list()
    
    for i in range(len(idx_lst)):
        coords_lst.append(np.asarray(corr_maps[i].shape) // 2)
        coords_lst.append((np.asarray(corr_maps[i].shape) // 2) + (1.7*idx_lst[i]))
    return coords_lst

def plot_image_witharrows(img, displ, w, h, filename=None):
    fig = pyplot.figure(figsize=(15, 15))
    pyplot.imshow(img, aspect='equal')
    cols = img.shape[1] // w
    cols_r = (img.shape[1] % w) // 2
    rows = img.shape[0] // h
    rows_r = (img.shape[0] % h) // 2
    k = 0
    l = 0
    ij = np.array( [w//2, h//2] )
    for d in displ:
        pyplot.annotate("", xy=d+ij, xytext=ij, 
                        arrowprops=dict(arrowstyle="->", color="orange", lw=3.))
        l = l+1
        if(l<cols): 
            ij[0] = (l*w) + (w//2)
        if(l==cols): 
            ij[0] = l*w + cols_r
        if(l > cols):
            ij[0] = w//2
            l = 0
            k = k+1
            if( k < rows):
                ij[1] = k*h + (h//2)
            if k==rows:
                ij[1] = k*h + (rows_r)
    if(filename != None):
        pyplot.savefig(filename, bbox_inches='tight')    

@jit
def calculate_displacements_process(zones1, zones2, my_num, processes_num, out_queue):
	# Calculate own number of zones to process
	residual = len(zones1) % processes_num
	N = len(zones1) // processes_num + (residual > my_num)
	# Calculate starting point
	if residual > my_num :
		start_idx = my_num * N
	else:
		start_idx = (residual)*(N+1) + (my_num - residual)*N
	# Now, the proper displacements computation
	# First the correlation maps
	corr_maps = compute_correlation_maps(zones1[start_idx : start_idx + N], zones2[start_idx : start_idx + N])
	# Get displacements from correlation maps
	max_idxs = get_displacements(corr_maps)
	# Save results to queue
	out_queue.put((my_num, max_idxs))

def main(args=None):
	if args is None:
   		args = sys.argv[1:]
	else:
		return

	start_time = time.time()

	# Load the two images
	img1 = load_image( sys.argv[1] )
	img2 = load_image( sys.argv[2] )
	# Define interrogation zones
	int_zones_w = int(sys.argv[3])
	int_zones_h = int(sys.argv[4])
	# Divide interrogation zones
	img1_int_zones, p_rows, p_cols = split_image(img1, int_zones_w, int_zones_h)
	img2_int_zones, p_rows, p_cols = split_image(img2, int_zones_w, int_zones_h)
	# Calculate displacements using processes

	start_time2 = time.time()

	output = mp.Queue()
	#processes = [mp.Process(target=calculate_displacements_process, args=(img1_int_zones, img2_int_zones, x, mp.cpu_count(), output)) for x in range(mp.cpu_count())]
	processes = [mp.Process(target=calculate_displacements_process, args=(img1_int_zones, img2_int_zones, x, 4, output)) for x in range(4)]
	for p in processes:
	    p.start()
	for p in processes:
	    p.join()

	start_time3 = time.time()
	
	# Now to recover the outputs
	displacements = [output.get() for p in processes]
	displacements.sort()
	#displacements = [d[1] for d in displacements]
	displacements = [inner for i in displacements for inner in i[1]]

	end_time = time.time()
	print("Cargar las imágenes y preparar las zonas de interrogación tomó ", start_time2 - start_time, "segundos")
	print("Calcular los desplazamientos tomó ", start_time3 - start_time2, "segundos")
	print("Recuperar los resultados desde la salida de los procesos tomó ", end_time - start_time3, "segundos")

	# Save the results
	plot_image_witharrows(img1, displacements, int_zones_w, int_zones_h, sys.argv[5])


if __name__ == '__main__':
	 main()
