# Import Libararies
import cv2
import sys
import os
import cv2
import numpy as np
import pandas as pd
#!pip install patchify
import patchify
import matplotlib.pyplot as plt
from numpy import arange
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
import numpy as np
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import scipy
from sklearn.linear_model import Lasso
import scipy.misc
import warnings
warnings.filterwarnings("ignore")


def OpenCV_imgRead(fileName):
    """
    load the input image into a matrix
    :param fileName: name of the input file
    :return: a matrix of the input image
    Examples: imgIn = imgRead('lena.bmp')
    """
    # Reading image
    imgIn = cv2.imread(fileName,cv2.IMREAD_UNCHANGED)
    #Dimensions of the image
    sizeX = imgIn.shape[1]
    sizeY = imgIn.shape[0]
    print('Image Height={},weight={}'.format(sizeY,sizeX))
    return imgIn


def imgShow(imgOut,cmap='gray'):
    """
    show the image saved in a matrix
    :param imgOut: a matrix containing the image to show
    :return: None
    """
    imgOut = np.uint8(imgOut)
    plt.imshow(imgOut,cmap=cmap)

def extract_image_blocks_using_patchify(img,patch_size,steps):
    """
    load the input image matrix and return image blocks of desired size
    :param
    img: input image matrix
    patch_size: desired block size
    step: In how many steps to extract patches
    return: An array of image blocks
    Examples: imgIn = imgRead('lena.bmp')
    """
    extracted_image_blocks = patchify.patchify(img, patch_size, step=steps)
    print('{} image blocks of size={}'.format(extracted_image_blocks.shape[0]*extracted_image_blocks.shape[1],patch_size))
    return extracted_image_blocks


def plot_blocks_and_save_img(image_blocks_numpy,save_image,save_img_name='patchesBoat.png'):
  subplot_number_of_colums = image_blocks_numpy.shape[1]
  subplot_number_of_rows   = image_blocks_numpy.shape[0]
  count=0
  # Subplot
  fig = plt.figure(figsize=(4,4))
    # loop through each of the subplot location and plot predictive Vs target variable

  for subplot_count_row in range(0,subplot_number_of_rows):
    for subplot_count_column in range(0,subplot_number_of_colums):
      count += 1

      ax = fig.add_subplot(subplot_number_of_rows, subplot_number_of_colums, count)
      # Using the Seaborn library function: seaborn.regplot () , ref: https://seaborn.pydata.org/generated/seaborn.regplot.html
      # This function plot data and a linear regression model fit.
      plt.imshow(image_blocks_numpy[subplot_count_row][subplot_count_column],'gray')
  plt.tight_layout()
  plt.subplots_adjust(wspace=0, hspace=0)
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
  if save_image=='True':
     plt.savefig(save_img_name,bbox_inches="tight",dpi=300)

  return


def get_random_pixel_sample(img_block,n_pixel_sample):
  # read oRandom_indexesginal image and downsize for speed
  X = img_block
  ny, nx = X.shape
  # extract small sample of signal
  Random_indexes = np.random.choice(nx * ny, n_pixel_sample, replace=False) # random sample of indices
  b = X.T.flat[Random_indexes]
  b = np.expand_dims(b, axis=1)
  # creating DCT matRandom_indexesx operator using kron
  dct_mat = np.kron(spfft.idct(np.identity(nx), norm='ortho', axis=0),spfft.idct(np.identity(ny), norm='ortho', axis=0))
  dct_mat = dct_mat[Random_indexes,:] # same as phi times kron
  return dct_mat,b,nx,ny

def CrossValidation_to_get_best_lemda_and_max_iter(x,y,n_split,n_repeats):
  # define model
  model = Lasso()
  # define model evaluation method
  cv = RepeatedKFold(n_splits=n_split, n_repeats=n_repeats, random_state=1)
  grid = dict()
  grid['alpha'] = np.logspace(-6, 6, 30, endpoint=False)
  grid['max_iter']  = [2000]
  # define search
  search = GridSearchCV(model, grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
  # perform the search
  results = search.fit(x,y)
  # summarize
  #print('MSE: %.3f' % results.best_score_)
  #print('Config: %s' % results.best_params_)
  return results.best_params_['alpha'],results.best_params_['max_iter']

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def Apply_Lasso_with_cv_pram_and_get_DCTcoeffient(x,y,best_alpha,best_max_iter):
  lasso = Lasso(alpha=best_alpha,max_iter=best_max_iter)
  lasso.fit(x, y)
  return lasso.coef_


def MSE_grayimg(img1, img2):
    squared_diff = (img1 -img2) ** 2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
    err = summed / num_pix
    return err

'''
img = OpenCV_imgRead('/content/drive/MyDrive/Duke-ECE-PHD/ECE-580-Intro_to_ML-Spring22/MP-1/nature.bmp')
imgShow(img)

patches_image = extract_image_blocks_using_patchify(img,patch_size=(16,16),steps=16)
plot_blocks_and_save_img(image_blocks_numpy=patches_image,save_image='False',save_img_name='patchesBoat.png')
'''


boat_img_PixelSamples = [10,20,30,40,50]
boat_img_BlockSize    = (8,8)
boat_img_Path         = 'fishing_boat.bmp'


nature_img_PixelSamples = [10,30,50,100,150]
nature_img_BlockSize    = (16,16)
nature_img_Path         = 'nature.bmp'


Compressed_Sensing_img = 'nature'
if Compressed_Sensing_img == 'boat':
    roi_img_PixelSamples = boat_img_PixelSamples
    roi_img_BlockSize    = boat_img_BlockSize
    roi_img_Path         = boat_img_Path
if Compressed_Sensing_img == 'nature':
    roi_img_PixelSamples = nature_img_PixelSamples
    roi_img_BlockSize    = nature_img_BlockSize
    roi_img_Path         = nature_img_Path



Sample_pixel_list              =[]
MSE_wO_Median_filter_list      =[]
MSE_w_Median_filter_list_scipy =[]
MSE_w_Median_filter_list_cv2   =[]


img                   = OpenCV_imgRead(roi_img_Path)
patches_image         = extract_image_blocks_using_patchify(img,patch_size=roi_img_BlockSize,steps=roi_img_BlockSize[0])

for Pixel_SamplePicked in range(0,len(roi_img_PixelSamples)):

    aplha_saving_matrix         = np.zeros((patches_image.shape[0],patches_image.shape[1]))
    reconstarcted_image_storage = np.zeros(patches_image.shape)
    blocks_number_of_colums = patches_image.shape[1]
    blocks_number_of_rows   = patches_image.shape[0]
    count=0
    for blocks_count_of_rows in range(0,blocks_number_of_rows):
        for blocks_count_of_colums in range(0,blocks_number_of_colums):
          count += 1

          roi_block = patches_image[blocks_count_of_rows][blocks_count_of_colums]
          dct_mat,b,nx,ny       = get_random_pixel_sample(img_block=roi_block,n_pixel_sample=roi_img_PixelSamples[Pixel_SamplePicked])
          cv_alpha,cv_max_iter  = CrossValidation_to_get_best_lemda_and_max_iter(x=dct_mat,y=b,n_split=6,n_repeats=20)
          Lasso_coefficeny      = Apply_Lasso_with_cv_pram_and_get_DCTcoeffient(x=dct_mat,y=b,best_alpha=cv_alpha,best_max_iter=cv_max_iter)
          Xat = np.array(Lasso_coefficeny).reshape(nx, ny).T # stack columns
          # Get the reconstructed image
          Xa = idct2(Xat)
          print(Lasso_coefficeny)
          print(cv_alpha)
          print(cv_max_iter)
          print('processing-block:{}, S={}'.format(count,roi_img_PixelSamples[Pixel_SamplePicked]))
          aplha_saving_matrix[blocks_count_of_rows][blocks_count_of_colums]         = cv_alpha
          reconstarcted_image_storage[blocks_count_of_rows][blocks_count_of_colums] = Xa

    reconstructed_image = patchify.unpatchify(reconstarcted_image_storage, img.shape)
    cv2.imwrite("{}-block_alpha_S{}.png".format(Compressed_Sensing_img,roi_img_PixelSamples[Pixel_SamplePicked]),reconstructed_image)
    np.savetxt("{}-block_alpha_S{}.csv".format(Compressed_Sensing_img,roi_img_PixelSamples[Pixel_SamplePicked]), aplha_saving_matrix, delimiter=",")

    Median_filter_reconstructed_img       = scipy.ndimage.median_filter(reconstructed_image, size=3)


    cv2.imwrite("{}-block_alpha_S{}_MedianFilter_scipy.png".format(Compressed_Sensing_img,roi_img_PixelSamples[Pixel_SamplePicked]),Median_filter_reconstructed_img)

    mse_without_median_filter          = MSE_grayimg(img, reconstructed_image)
    mse_with_median_filter_scipy       = MSE_grayimg(img, Median_filter_reconstructed_img)


    Sample_pixel_list.append(roi_img_PixelSamples[Pixel_SamplePicked])
    MSE_wO_Median_filter_list.append(mse_without_median_filter)
    MSE_w_Median_filter_list_scipy.append(mse_with_median_filter_scipy)



infoDatafeame= pd.DataFrame(list(zip(Sample_pixel_list,MSE_wO_Median_filter_list,MSE_w_Median_filter_list_scipy)),columns=['Sample','MSE-WO-Median_filter','MSE-W-Median_filter'])
infoDatafeame.to_csv("{}-image_results_analysis.csv".format(Compressed_Sensing_img),encoding='utf-8',index=False)




#plot_blocks_and_save_img(image_blocks_numpy=patches_image,save_image='False',save_img_name='fishing_boat_block.png')
#plot_blocks_and_save_img(image_blocks_numpy=patches_image,save_image='False',save_img_name='fishing_boat_block.png')
